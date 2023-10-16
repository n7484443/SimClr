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

#참조 : https://github.com/p3i0t/SimCLR-CIFAR10/tree/master

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
        crop_transform = transforms.RandomResizedCrop((32, 32), scale=(0.6, 1), ratio=(0.5, 2.0), )
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

# 출처 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
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
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, projection_dim, bias=False)
        )

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection


if __name__ == '__main__':
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 512
    hyper_batch_size_predictor = 512
    hyper_epoch = 50
    hyper_epoch_predictor = 50
    lr = 0.075*math.sqrt(hyper_batch_size)
    lr_predictor = 0.2 * (0.1 * hyper_batch_size_predictor / 256)
    momentum = 0.9
    temperature = 0.5
    strength = 0.9

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    feature_dim = 128
    class_size = 10
    simclr = SimCLR(base_encoder=torchvision.models.resnet18, projection_dim=feature_dim).to(device)
    simclr.train()

    loss_function = SimclrLoss(temperature=temperature).to(device)

    summary(simclr, input_size=(hyper_batch_size, 3, 32, 32))

    optimizer = optim.LARS(simclr.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainLoader), eta_min=0, last_epoch=-1)

    distortion = Distortion(strength=strength).to(device)
    # train
    data_output = []
    for epoch in range(hyper_epoch):
        first = False
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        for batch_data, _ in trainLoader:
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

            _, batch_data_after = simclr.forward(batch_data_cuda)
            _, batch_distorted_after = simclr.forward(batch_data_distorted)

            loss = loss_function.forward(batch_data_after, batch_distorted_after, batch_size)
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()

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
        print('Epoch : %d, Avg Loss : %.4f, Validation Loss : %.4f Leraning Late: %.4f' % (
            epoch, loss_sum_train, loss_sum_test, scheduler.get_last_lr()[0]))
        data_output.append((loss_sum_train, loss_sum_test))
        if epoch >= 10:
            scheduler.step()

    plt.subplot(2, 1, 1)
    range_x = np.arange(0, hyper_epoch, 1)
    plt.plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    plt.plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    plt.yscale('log')
    plt.legend()
    plt.title(f'lr: {lr} batch:{hyper_batch_size} epoch: {hyper_epoch} temp:{temperature}')

    print("FG 학습 완료. 이제 F의 output을 실제 dataset의 label과 연결.")

    trainLoader, testLoader = load_image(batch_size=hyper_batch_size_predictor)
    # predictor
    predictor = nn.Linear(simclr.feature_dim, class_size).to(device)
    # fg output 과 실제 dataset label의 연결
    simple_loss_function = nn.CrossEntropyLoss().to(device)

    simclr.eval()
    predictor.train()
    optimizer = optim.LARS(predictor.parameters(), lr=lr_predictor, momentum=momentum)
    data_output = []
    for epoch in range(hyper_epoch_predictor):
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        loss_sum_test = 0
        for batch_data, batch_label in trainLoader:
            batch_size = batch_data.shape[0]

            with torch.no_grad():
                feature, _ = simclr.forward(batch_data.to(device))
            expect = predictor.forward(feature)
            loss = simple_loss_function(expect, batch_label.to(device))
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for batch_data, batch_label in testLoader:
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                feature, _ = simclr.forward(batch_data_cuda)
                expect = predictor.forward(feature)
                loss = simple_loss_function(expect, batch_label.to(device))
                loss_sum_test += loss.item()

        loss_sum_train /= len(trainLoader)
        loss_sum_test /= len(testLoader)
        print('Epoch : %d, Avg Loss : %.4f Validation Loss : %.4f' % (
            epoch, loss_sum_train, loss_sum_test))
        data_output.append((loss_sum_train, loss_sum_test))

    plt.subplot(2, 1, 2)
    range_x = np.arange(0, hyper_epoch_predictor, 1)
    plt.plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    plt.plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    plt.yscale('log')
    plt.legend()
    plt.title(f'predictor \nlr: {lr_predictor} batch:{hyper_batch_size_predictor} epoch: {hyper_epoch_predictor}')

    plt.tight_layout()
    plt.show()

    predictor.eval()

    print("실제 test")
    correct_1 = 0
    correct_5 = 0
    total = 10000
    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            feature, _ = simclr.forward(batch_data.to(device))
            output = predictor.forward(feature)
            # argmax = 가장 큰 값의 index를 가져옴
            _, predicted_1 = torch.topk(output, k=1, dim=1)
            _, predicted_5 = torch.topk(output, k=5, dim=1)
            correct_1 += torch.eq(predicted_1, batch_label.to(device).view([-1, 1])).any(dim=1).sum().item()
            correct_5 += torch.eq(predicted_5, batch_label.to(device).view([-1, 1])).any(dim=1).sum().item()
    print(f'총 개수 : {total}')
    print(f'top-1 맞춘 개수 : {correct_1} \n 정확도: {100.0 * correct_1 / total}')
    print(f'top-5 맞춘 개수 : {correct_5} \n 정확도: {100.0 * correct_5 / total}')
