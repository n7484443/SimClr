import math

import torch.utils.data as tu_data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch_optimizer as optim
from torchinfo import summary
import numpy as np
from tqdm import tqdm
from time import sleep


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, **kwargs):
        super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        # 일반적으로 base_optimizer을 기반으로 Sharpness가 작은 부분으로 최적화
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self):
        device_ = self.param_groups[0]["params"][
            0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device_)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    # loss 의 최적화
    # epsilon hat 의 근사 계산, w + epsilon hat 으로 포인트 이동
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # p 텐서 꼴로 변환 후 gradient 곱하기
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # local maximum "w + e_hat"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    # loss sharpness 의 최적화
    # epsilon hat 을 빼서 기존 지점으로 원복후 파라미터 업데이트
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss.item()


# 참조 : https://github.com/p3i0t/SimCLR-CIFAR10/tree/master
# 참조 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # pin_memory : GPU에 데이터를 미리 전송
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # validation set을 분리하는게 맞으나, 여러 모델을 테스트하고 제일 좋은걸 선별하는 게 아니므로 test set을 validation set으로 사용
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# 이미지 텐서를 색상 변환, 크롭
class Distortion(nn.Module):
    def __init__(self, strength, image_size=32):
        super().__init__()
        crop_transform = transforms.RandomResizedCrop((image_size, image_size), scale=(0.25, 1),
                                                      ratio=(1.0 / 2.0, 2.0 / 1.0), antialias=True)
        flip_horizon = transforms.RandomHorizontalFlip(p=0.5)
        flip_vertical = transforms.RandomVerticalFlip(p=0.5)
        color_transform = transforms.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength,
                                                 saturation=0.8 * strength, hue=0.2 * strength)
        self.transform_several = torch.nn.Sequential(
            crop_transform,
            flip_vertical,
            flip_horizon,
            color_transform,
        ).to(device)

    def forward(self, x):
        return self.transform_several(x)


class SimclrLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # loss 분모 부분의 negative sample 간의 내적 합만을 가져오기 위한 마스킹 행렬
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본 - augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(weights=None)
        self.feature_dim = self.enc.fc.in_features

        self.enc.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()

        self.projection_dim = projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.feature_dim, projection_dim, bias=False)
        )

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection


if __name__ == '__main__':
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 512
    hyper_batch_size_predictor = 128
    hyper_epoch = 100
    hyper_epoch_predictor = hyper_epoch*2
    lr = 0.075 * math.sqrt(hyper_batch_size)
    lr_predictor = 1e-3
    weight_decay_predictor = 1e-6
    temperature = 0.1
    strength = 1

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    projection_dim = 128
    class_size = 10
    simclr = SimCLR(base_encoder=torchvision.models.resnet18, projection_dim=projection_dim).to(device)
    simclr.train()

    loss_function = SimclrLoss(temperature=temperature).to(device)

    summary(simclr, input_size=(hyper_batch_size, 3, 32, 32))

    optimizer = SAM(simclr.parameters(), torch.optim.Adam, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_epoch - 10, eta_min=0,
                                                           last_epoch=-1)

    distortion = Distortion(strength=strength).to(device)
    distortion.eval()
    # train
    data_output = []
    for epoch in range(1, hyper_epoch + 1):
        first = False
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        tqdm_epoch = tqdm(trainLoader, unit="batch")
        for batch_data, _ in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch}")
            with torch.no_grad():
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_distorted = distortion.forward(batch_data_cuda)
                if first:
                    first = False
                    pil_img = transforms.ToPILImage()(batch_data_cuda[0])
                    plt.imshow(pil_img)
                    plt.show()
                    pil_img2 = transforms.ToPILImage()(batch_data_distorted[0])
                    plt.imshow(pil_img2)
                    plt.show()
                # [batch_size * 2, rgb, width, height] => [batch_size * 2, h_value_size]


            def closure():
                optimizer.zero_grad()
                _, batch_data_after = simclr.forward(batch_data_cuda)
                _, batch_distorted_after = simclr.forward(batch_data_distorted)
                loss = loss_function.forward(batch_data_after, batch_distorted_after, batch_size)
                loss.backward()
                return loss


            loss_sum_train += optimizer.step(closure)
        tqdm_epoch.close()
        loss_sum_test = 0
        with torch.no_grad():
            for batch_data, _ in testLoader:
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_distorted = distortion.forward(batch_data_cuda)
                _, batch_data_after = simclr.forward(batch_data_cuda)
                _, batch_distorted_after = simclr.forward(batch_data_distorted)
                loss = loss_function.forward(batch_data_after, batch_distorted_after, batch_size)
                loss_sum_test += loss.item()

        loss_sum_train /= len(trainLoader)
        loss_sum_test /= len(testLoader)
        tqdm.write('Avg Loss : %.4f Validation Loss : %.4f Learning Late: %.4f' % (
            loss_sum_train, loss_sum_test, scheduler.get_last_lr()[0]))
        sleep(0.1)
        data_output.append((loss_sum_train, loss_sum_test))
        if epoch >= 10:
            scheduler.step()

    fig, ax = plt.subplots(2, 1)
    range_x = np.arange(0, hyper_epoch, 1)
    ax[0].plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    ax[0].plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_title(f'lr: {lr} batch:{hyper_batch_size} epoch: {hyper_epoch} temp:{temperature}')

    print("FG 학습 완료. 이제 F의 output을 실제 dataset의 label과 연결.")

    trainLoader, testLoader = load_image(batch_size=hyper_batch_size_predictor)
    # predictor
    predictor = nn.Linear(simclr.feature_dim, class_size).to(device)
    # fg output_resnet18 과 실제 dataset label의 연결
    simple_loss_function = nn.CrossEntropyLoss().to(device)

    simclr.eval()
    predictor.train()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr_predictor, weight_decay=weight_decay_predictor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_epoch_predictor - 10, eta_min=0,
                                                           last_epoch=-1)
    data_output = []
    for epoch in range(1, hyper_epoch_predictor + 1):
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        loss_sum_test = 0
        total_size = 0
        tqdm_epoch = tqdm(trainLoader, unit="batch")
        for batch_data, batch_label in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch}")
            batch_size = batch_data.shape[0]

            with torch.no_grad():
                feature, _ = simclr.forward(batch_data.to(device))
            expect = predictor.forward(feature)
            loss = simple_loss_function(expect, batch_label.to(device))
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()
        tqdm_epoch.close()
        correct = 0.0
        with torch.no_grad():
            for batch_data, batch_label in testLoader:
                batch_label_cuda = batch_label.to(device)
                batch_size = batch_data.size(dim=0)
                batch_data_cuda = batch_data.to(device)
                feature, _ = simclr.forward(batch_data_cuda)
                expect = predictor.forward(feature)
                _, predicted_1 = torch.topk(expect, k=1, dim=1)
                correct += torch.eq(predicted_1, batch_label_cuda.view([-1, 1])).any(dim=1).sum().item()
                loss = simple_loss_function(expect, batch_label_cuda)
                loss_sum_test += loss.item()
                total_size += batch_size

        loss_sum_train /= len(trainLoader)
        loss_sum_test /= len(testLoader)
        data_output.append((loss_sum_train, loss_sum_test, 100.0 * correct / total_size))
        tqdm.write('Avg Loss : %.4f Validation Loss : %.4f Learning Late: %.4f' % (
            loss_sum_train, loss_sum_test, scheduler.get_last_lr()[0]))
        sleep(0.1)
        if epoch >= 10:
            scheduler.step()

    range_x = np.arange(0, hyper_epoch_predictor, 1)
    ax[1].plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    ax[1].plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    ax2 = ax[1].twinx()
    ax2.plot(range_x, [x[2] for x in data_output], label='Accuracy', color='green')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_title(f'predictor \nlr: {lr_predictor} batch:{hyper_batch_size_predictor} epoch: {hyper_epoch_predictor}')

    plt.tight_layout()
    plt.show()

    predictor.eval()

    print("실제 test")
    correct_1 = 0
    correct_5 = 0
    total_size = 0
    with torch.no_grad():
        tqdm_epoch = tqdm(testLoader, unit="batch")
        for batch_data, batch_label in tqdm_epoch:
            feature, _ = simclr.forward(batch_data.to(device))
            output = predictor.forward(feature)
            # argmax = 가장 큰 값의 index를 가져옴
            _, predicted_1 = torch.topk(output, k=1, dim=1)
            _, predicted_5 = torch.topk(output, k=5, dim=1)
            correct_1 += torch.eq(predicted_1, batch_label.to(device).view([-1, 1])).any(dim=1).sum().item()
            correct_5 += torch.eq(predicted_5, batch_label.to(device).view([-1, 1])).any(dim=1).sum().item()
            total_size += batch_data.size(dim=0)
        tqdm_epoch.close()
    print(f'총 개수 : {total_size}')
    print(f'top-1 맞춘 개수 : {correct_1} \n 정확도: {100.0 * correct_1 / total_size}')
    print(f'top-5 맞춘 개수 : {correct_5} \n 정확도: {100.0 * correct_5 / total_size}')
