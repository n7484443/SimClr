import math
import os

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
from umap import UMAP


# 참조 : https://github.com/p3i0t/SimCLR-CIFAR10/tree/master
# 참조 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # pin_memory : GPU에 데이터를 미리 전송
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # validation set을 분리하는게 맞으나, 여러 모델을 테스트하고 제일 좋은걸 선별하는 게 아니므로 test set을 validation set으로 사용
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

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

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.int64)

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


def learning_resnet(model, hyper_epoch, device, lr, temperature, strength, trainLoader, testLoader, hyper_batch_size, weight_decay):
    loss_function = SimclrLoss(temperature=temperature).to(device)

    distortion = Distortion(strength=strength).to(device)
    distortion.eval()
    # train
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000 - 10, eta_min=0,
                                                           last_epoch=-1)

    model.eval()
    actual = []
    deep_features = []

    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            batch_data_cuda = batch_data.to(device)
            feature, _ = model(batch_data_cuda)

            deep_features += feature.cpu().numpy().tolist()
            actual += batch_label.cpu().numpy().tolist()
    umap = UMAP(n_components=2)
    cluster = np.array(umap.fit_transform(np.array(deep_features)))
    actual = np.array(actual)

    plt.figure(figsize=(10, 10))
    cifar = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i, label in zip(range(10), cifar):
        idx = np.where(actual == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
    plt.legend()
    plt.show()


    data_output = []
    for epoch in range(1, hyper_epoch + 1):
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        tqdm_epoch = tqdm(trainLoader, unit="batch")
        model.train()
        for batch_data, _ in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch}")
            with torch.no_grad():
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_distorted = distortion(batch_data_cuda)
                batch_input = torch.cat((batch_data_cuda, batch_data_distorted), dim=0)

            optimizer.zero_grad()

            _, batch_input_after = model(batch_input)
            batch_data_after, batch_distorted_after = torch.chunk(batch_input_after, 2, dim=0)
            loss = loss_function(batch_data_after, batch_distorted_after, batch_size)
            loss.backward()
            optimizer.step()
            loss_sum_train += loss.item()
        tqdm_epoch.close()

        loss_sum_train /= len(trainLoader)
        tqdm.write('Avg Loss : %.4f Learning Late: %.4f' % (
            loss_sum_train, scheduler.get_last_lr()[0]))
        sleep(0.1)
        data_output.append(loss_sum_train)
        if epoch >= 10:
            scheduler.step()
        if epoch % 20 == 0:
            model.eval()
            actual = []
            deep_features = []

            with torch.no_grad():
                for batch_data, batch_label in testLoader:
                    batch_data_cuda = batch_data.to(device)
                    feature, _ = model(batch_data_cuda)

                    deep_features += feature.cpu().numpy().tolist()
                    actual += batch_label.cpu().numpy().tolist()
            cluster = np.array(umap.fit_transform(np.array(deep_features)))
            actual = np.array(actual)

            plt.figure(figsize=(10, 10))
            for i, label in zip(range(10), cifar):
                idx = np.where(actual == i)
                plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
            plt.legend()
            plt.show()
    torch.save(model.state_dict(), f"./model_sgd_{hyper_batch_size}_{lr}_100.pt")

    return data_output


def learning_predictor(model, model_predictor, hyper_epoch, device, lr, weight_decay, trainLoader, testLoader):
    simple_loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_epoch - 10, eta_min=0,
                                                           last_epoch=-1)
    data_output = []
    model.eval()
    for epoch in range(1, hyper_epoch + 1):
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        loss_sum_test = 0
        total_size = 0
        tqdm_epoch = tqdm(trainLoader, unit="batch")
        model_predictor.train()
        for batch_data, batch_label in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch}")

            with torch.no_grad():
                feature, _ = model(batch_data.to(device))
            expect = model_predictor(feature)
            loss = simple_loss_function(expect, batch_label.to(device))
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()
        tqdm_epoch.close()
        correct = 0.0
        model_predictor.eval()
        with torch.no_grad():
            for batch_data, batch_label in testLoader:
                batch_label_cuda = batch_label.to(device)
                batch_size = batch_data.size(dim=0)
                batch_data_cuda = batch_data.to(device)
                feature, _ = model(batch_data_cuda)
                expect = model_predictor(feature)
                _, predicted_1 = torch.topk(expect, k=1, dim=1)
                correct += torch.eq(predicted_1, batch_label_cuda.view([-1, 1])).any(dim=1).sum().item()
                loss = simple_loss_function(expect, batch_label_cuda)
                loss_sum_test += loss.item()
                total_size += batch_size

        loss_sum_train /= len(trainLoader)
        loss_sum_test /= len(testLoader)
        accuracy = 100.0 * correct / total_size
        data_output.append((loss_sum_train, loss_sum_test, accuracy))
        tqdm.write('Avg Loss : %.4f Validation Loss : %.4f Learning Late: %.4f Accuracy: %.4f' % (
            loss_sum_train, loss_sum_test, scheduler.get_last_lr()[0], accuracy))
        sleep(0.1)
        if epoch >= 10:
            scheduler.step()

    return data_output


if __name__ == '__main__':
    device = torch.device("cuda")
    hyper_batch_size = 128
    hyper_epoch = 100
    hyper_epoch_predictor = hyper_epoch
    lr = round(0.075 * math.sqrt(hyper_batch_size), 6)
    lr_predictor = 1e-3 * hyper_batch_size/64
    weight_decay = 1e-6
    temperature = 0.1
    strength = 1
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    projection_dim = 128
    class_size = 10
    continue_learning = True
    simclr = SimCLR(base_encoder=torchvision.models.resnet34, projection_dim=projection_dim).to(device)
    predictor = nn.Linear(simclr.feature_dim, class_size).to(device)
    if os.path.isfile(f"./model_sgd_{hyper_batch_size}_{lr}_100.pt"):
        simclr.load_state_dict(torch.load(f"./model_sgd_{hyper_batch_size}_{lr}_100.pt"))
    if continue_learning:
        summary(simclr, input_size=(hyper_batch_size, 3, 32, 32))

        data_output = learning_resnet(simclr, hyper_epoch, device=device, lr=lr, temperature=temperature,
                                      strength=strength, trainLoader=trainLoader, testLoader=testLoader,
                                      hyper_batch_size=hyper_batch_size, weight_decay=weight_decay)
        fig, ax = plt.subplots(2, 1)
        range_x = np.arange(0, hyper_epoch, 1)
        ax[0].plot(range_x, data_output, label='Training loss', color='red')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[0].set_title(f'lr: {lr} batch:{hyper_batch_size} epoch: {hyper_epoch} temp:{temperature}')

        print("FG 학습 완료. 이제 F의 output을 실제 dataset의 label과 연결.")

    data_output = learning_predictor(simclr, predictor, hyper_epoch_predictor, device=device, lr=lr_predictor,
                                     weight_decay=weight_decay, trainLoader=trainLoader,
                                     testLoader=testLoader)

    range_x = np.arange(0, hyper_epoch_predictor, 1)
    ax[1].plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    ax[1].plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    ax2 = ax[1].twinx()
    ax2.plot(range_x, [x[2] for x in data_output], label='Accuracy', color='green')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_title(f'predictor \nlr: {lr_predictor} batch:{hyper_batch_size} epoch: {hyper_epoch_predictor}')

    plt.tight_layout()
    plt.show()

    predictor.eval()

    print("실제 test")
    correct_1 = 0
    correct_5 = 0
    total_size = 0

    simclr.eval()
    predictor.eval()
    with torch.no_grad():
        tqdm_epoch = tqdm(testLoader, unit="batch")
        for batch_data, batch_label in tqdm_epoch:
            feature, _ = simclr(batch_data.to(device))
            output = predictor(feature)
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
