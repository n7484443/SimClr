import os

import torch.utils.data as tu_data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import Tensor
from torchinfo import summary


def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # X => (X - mean)/standard_deviations (정규분포의 Normalization)
    # 이미지는 일반적으로 0~255 혹은 0~1, 여기선 0~1 의 값
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # num_worker 2 이상은 windows에서 작동 안함(아마 버그)
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loader

# 이미지 텐서를 색상 변환, 크롭
def distortion(tensor_image: Tensor, strength: int) -> Tensor:
    # [batch_size, rgb, width, height]
    img_size = tensor_image.shape[2]
    color_transform = transforms.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength,
                                             saturation=0.8 * strength, hue=0.2 * strength)
    # 0.1~1 사이의 크기, 0.5~2 사이의 종횡비로 랜덤하게 크롭
    crop_transform = transforms.RandomResizedCrop((img_size, img_size), scale=(0.5, 1), ratio=(0.5, 2), )
    flip_horizon = transforms.RandomHorizontalFlip(p=0.5)
    flip_vertical = transforms.RandomVerticalFlip(p=0.5)
    transform_several = torch.nn.Sequential(
        crop_transform,
        flip_vertical,
        flip_horizon,
        color_transform,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    return transform_several.forward(tensor_image)


# identity resnet 구현
class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.sequentialLayer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, bias=False, padding='same'),
            nn.BatchNorm2d(out_channel)
        )
        if in_channel != out_channel:
            # 1x1 conv, 크기 조절을 위한 projection mapping
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sequentialLayer(x) + self.shortcut(x)
        x = self.relu(x)
        return x


# conv output size = ceil((i + 2p - k) / s) + 1
class Resnet(nn.Module):
    def __init__(self, size):
        super().__init__()
        # rgb 3개로 64개의 채널 만듬
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
        # 64 * image width * image height
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            *[ResnetBlock(64, 64) for _ in range(2)],
        )
        # 128 * image width * image height
        self.conv3 = nn.Sequential(
            ResnetBlock(64, 128),
            *[ResnetBlock(128, 128) for _ in range(2)]
        )
        # 전체 평균, 즉 512 * 1 * 1
        # AvgPool2d 는 Conv2d와 같이 커널을 이동시키며 평균을 구함
        # AdaptiveAvgPool2d 는 AvgPool2d와 비슷하나 특정 크기로 자동으로 맞춰줌
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer로 512 * 1 * 1 을 size 크기로 변환
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, size)
        )

    def forward(self, x):
        # 2048 * 3 * 32 * 32 -> 2048 * 64 * 16 * 16
        x = self.conv1(x)

        # -> 2048 * 64 * 16 * 16
        x = self.conv2(x)

        # -> 2048 * 128 * 8 * 8
        x = self.conv3(x)

        # 2048 * 128 * 2 * 2 -> 2048 * 128 * 1 * 1
        x = self.avg_pool(x)

        # 2048 * 128 * 1 * 1 -> 2048 * 128 -> 2048 * size
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    # SimCLR에 사용된 최소화된 MLP
    def __init__(self, size):
        super().__init__()
        self.hidden = nn.Linear(size, size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return x


class SimCLRLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cosine = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, sample_size, device):
        x = nn.functional.normalize(x, dim=1)
        # print(x)
        # similarity 계산
        x = torch.mm(x, torch.transpose(x, 0, 1)) / self.temp
        # print(x)
        # 대각 부분은 제외되어야 함
        # 즉 대각 부분만 추출 -> 다시 대각 행렬로 변환하여 마스킹 가능
        x = x - torch.diag(torch.diag(x, 0))
        # print(x)
        # down[0] -> up[0] -> down[2] -> up[2] -> 형식
        # 즉 대각 성분을 2칸씩 건너뛰며 추출
        mask = torch.tensor([(i + 1) % 2 for i in range(sample_size - 1)], dtype=torch.float, device=device)
        up_mask = torch.diag(mask, 1)
        down_mask = torch.diag(mask, -1)
        mask = up_mask + down_mask
        # print(mask)

        masked_x = x * mask
        masked_x = torch.sum(masked_x, dim=1)
        masked_x = torch.exp(masked_x)
        # print(masked_x)
        e_x = torch.exp(x)
        e_x = e_x - torch.eye(sample_size, device=device)
        # print(e_x)
        e_x = torch.sum(e_x, dim=1)
        # print(e_x)
        output = torch.div(masked_x, e_x)
        # print(output)
        output = -torch.log(output)
        # print(masked_x, e_x, torch.sum(output) / sample_size)
        return torch.sum(output) / sample_size


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 512

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    size = 32
    f_resnet = Resnet(size=size)
    g_small = SimpleMLP(size=size)
    loss_function = SimCLRLoss(temp=0.5)
    fg = nn.Sequential(f_resnet, g_small)

    summary(fg, input_size=(hyper_batch_size, 3, 32, 32))

    fg.to(device)
    if os.path.exists('./model.pt'):
        fg = torch.load('./model.pt')
    if not os.path.exists('./model.pt') or want_train:
        fg.train()
        optimizer = torch.optim.Adam(fg.parameters(), lr=0.001)
        # train
        for epoch in range(40):
            first = True
            # batch_size * 2, rgb, x, y 의 데이터 형태
            loss_sum = 0
            for batch_data, batch_label in trainLoader:
                batch_data_distorted = distortion(tensor_image=batch_data, strength=1)
                batch_size = batch_data.shape[0]
                batch = torch.zeros((batch_size * 2, 3, size, size), device=device)
                batch[::2, :] = batch_data.to(device)
                batch[1::2, :] = batch_data_distorted
                # print(batch.shape)
                # print(batch_label.shape)
                if first:
                    first = False
                    pil_img = transforms.ToPILImage()(batch[0])
                    plt.imshow(pil_img)
                    plt.show()
                    pil_img2 = transforms.ToPILImage()(batch[1])
                    plt.imshow(pil_img2)
                    plt.show()
                # [batch_size * 2, rgb, width, height] => [batch_size * 2, h_value_size]
                loss = loss_function.forward(fg.forward(batch), batch_size * 2, device)
                loss_sum += loss.item()

                optimizer.zero_grad()  # gradient 초기화
                loss.backward()
                optimizer.step()
                # print(loss)

            print('Epoch : %d, Avg Loss : %.4f' % (epoch, loss_sum / len(trainLoader)))
        torch.save(fg, './model.pt')
    fg.eval()
    fg.to(device)

    print("FG 학습 완료. 이제 FG의 output을 실제 dataset의 label과 연결.")

    if os.path.exists('./fg_output.pt'):
        simple_mlp = torch.load('./fg_output.pt')
    if not os.path.exists('./fg_output.pt') or want_train:
        # fg output 과 실제 dataset label의 연결
        simple_mlp = nn.Sequential(
            nn.Linear(size, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 10)
            # class는 10개
        )
        simple_loss_function = nn.CrossEntropyLoss()
        simple_loss_function.to(device)
        simple_mlp.to(device)
        simple_mlp.train()
        optimizer = torch.optim.Adam(simple_mlp.parameters(), lr=0.001)
        for epoch in range(40):
            # batch_size * 2, rgb, x, y 의 데이터 형태
            loss_sum = 0
            for batch_data, batch_label in trainLoader:
                batch_size = batch_data.shape[0]
                output = fg.forward(batch_data.to(device))
                expect = simple_mlp.forward(output)
                loss = simple_loss_function(expect, batch_label.to(device))
                loss_sum += loss.item()

                optimizer.zero_grad()  # gradient 초기화
                loss.backward()
                optimizer.step()
                # print(loss)

            print('Epoch : %d, Avg Loss : %.4f' % (epoch, loss_sum / len(trainLoader)))
        torch.save(fg, './fg_output.pt')
    else:
        simple_mlp = torch.load('./fg_output.pt')
    simple_mlp.eval()
    simple_mlp.to(device)

    print("실제 test")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            output = fg.forward(batch_data.to(device))
            output = simple_mlp.forward(output)
            _, predicted = torch.max(output.data, 1)
            total += batch_label.size(0)
            correct += (predicted == batch_label.to(device)).sum().item()
    print('총 개수 : %i \n 맞춘 개수 : %i \n 정확도: %d \n 찍었을 때의 정확도 : %d' % (total, correct, 100.0 * correct / total, 10))
