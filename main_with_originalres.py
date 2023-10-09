import os

import torch.utils.data as tu_data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import Tensor
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # X => (X - mean)/standard_deviations (정규분포의 Normalization)
    # 이미지는 일반적으로 0~255 혹은 0~1, 여기선 0~1 의 값
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # num_worker 2 이상은 windows에서 작동 안함(아마 버그)
    # pin_memory : GPU에 데이터를 미리 전송
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    return train_loader, test_loader


# 이미지 텐서를 색상 변환, 크롭
def distortion(tensor_image: Tensor, strength: int) -> Tensor:
    # [batch_size, rgb, width, height]
    img_size = tensor_image.shape[2]
    color_transform = transforms.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength,
                                             saturation=0.8 * strength, hue=0.2 * strength)
    # 0.5~1 사이의 크기, 0.5~2 사이의 종횡비로 랜덤하게 크롭
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


class SimpleMLP(nn.Module):
    # SimCLR에 사용된 최소화된 MLP
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, output_size)
        self.hidden = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.relu(x)
        return x


# 출처 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
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

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본 - augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


#
# class SimCLRLoss(nn.Module):
#     def __init__(self, temp):
#         super().__init__()
#         self.temp = temp
#         self.cosine = torch.nn.CosineSimilarity(dim=1)
#
#     def forward(self, x, sample_size, device):
#         x = nn.functional.normalize(x, dim=1)
#         # print(x)
#         # similarity 계산
#         x = torch.mm(x, torch.transpose(x, 0, 1)) / self.temp
#         # print(x)
#         # 대각 부분은 제외되어야 함
#         # 즉 대각 부분만 추출 -> 다시 대각 행렬로 변환하여 마스킹 가능
#         x = x - torch.diag(torch.diag(x, 0))
#         # print(x)
#         # down[0] -> up[0] -> down[2] -> up[2] -> 형식
#         # 즉 대각 성분을 2칸씩 건너뛰며 추출
#         mask = torch.tensor([(i + 1) % 2 for i in range(sample_size - 1)], dtype=torch.float, device=device)
#         up_mask = torch.diag(mask, 1)
#         down_mask = torch.diag(mask, -1)
#         mask = up_mask + down_mask
#         # print(mask)
#
#         masked_x = x * mask
#         masked_x = torch.sum(masked_x, dim=1)
#         masked_x = torch.exp(masked_x)
#         # print(masked_x)
#         e_x = torch.exp(x)
#         e_x = e_x - torch.eye(sample_size, device=device)
#         # print(e_x)
#         e_x = torch.sum(e_x, dim=1)
#         # print(e_x)
#         output = torch.div(masked_x, e_x)
#         # print(output)
#         output = -torch.log(output)
#         # print(masked_x, e_x, torch.sum(output) / sample_size)
#         return torch.sum(output) / sample_size


if __name__ == '__main__':
    writer = SummaryWriter()
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 500
    hyper_epoch = 50

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    size = 32
    output_size = 10
    f_resnet = torchvision.models.resnet18(pretrained=False)
    num_input = f_resnet.fc.in_features
    f_resnet.fc = nn.Linear(num_input, size)
    f_resnet.to(device)

    loss_function = SimCLR_Loss(batch_size=hyper_batch_size, temperature=0.1)

    summary(f_resnet, input_size=(hyper_batch_size, 3, 32, 32))

    # if os.path.exists('./model.pt'):
        # f_resnet = torch.load('./model.pt')
    if not os.path.exists('./model.pt') or False:
        g_small = SimpleMLP(input_size=size, output_size=output_size)
        g_small.to(device)
        fg = nn.Sequential(
            f_resnet,
            g_small
        )
        fg.train()
        optimizer = torch.optim.Adam(fg.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # train
        for epoch in range(hyper_epoch):
            first = False
            # batch_size * 2, rgb, x, y 의 데이터 형태
            loss_sum = 0
            for batch_data, batch_label in trainLoader:
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_distorted = distortion(tensor_image=batch_data_cuda, strength=1)
                # print(batch.shape)
                # print(batch_label.shape)
                if first:
                    first = False
                    pil_img = transforms.ToPILImage()(batch_data_cuda[0])
                    plt.imshow(pil_img)
                    plt.show()
                    pil_img2 = transforms.ToPILImage()(batch_data_distorted[0])
                    plt.imshow(pil_img2)
                    plt.show()
                # [batch_size * 2, rgb, width, height] => [batch_size * 2, h_value_size]
                batch_data_after = fg.forward(batch_data_cuda)
                batch_distorted_after = fg.forward(batch_data_distorted)

                loss = loss_function.forward(batch_data_after, batch_distorted_after)
                loss_sum += loss.item()

                optimizer.zero_grad()  # gradient 초기화
                loss.backward()
                optimizer.step()
                # print(loss)

            print('Epoch : %d, Avg Loss : %.4f lr: %.8f' % (epoch, loss_sum / len(trainLoader), optimizer.param_groups[0]['lr']))
            scheduler.step()
        torch.save(f_resnet, './model.pt')
        writer.flush()
        writer.close()
    f_resnet.eval()
    f_resnet.to(device)

    print("FG 학습 완료. 이제 F의 output을 실제 dataset의 label과 연결.")

    # predictor
    predictor = nn.Sequential(
        nn.Linear(size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        # class는 10개
    )
    if os.path.exists('./fg_output.pt'):
        predictor = torch.load('./fg_output.pt')
    predictor.to(device)
    if not os.path.exists('./fg_output.pt') or want_train:
        # fg output 과 실제 dataset label의 연결
        simple_loss_function = nn.CrossEntropyLoss()
        simple_loss_function.to(device)
        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        for epoch in range(hyper_epoch):
            # batch_size * 2, rgb, x, y 의 데이터 형태
            loss_sum = 0
            for batch_data, batch_label in trainLoader:
                batch_size = batch_data.shape[0]
                output = f_resnet.forward(batch_data.to(device))
                expect = predictor.forward(output)
                loss = simple_loss_function(expect, batch_label.to(device))
                loss_sum += loss.item()

                optimizer.zero_grad()  # gradient 초기화
                loss.backward()
                optimizer.step()
                # print(loss)

            print('Epoch : %d, Avg Loss : %.4f lr: %.8f' % (epoch, loss_sum / len(trainLoader), optimizer.param_groups[0]['lr']))
            scheduler.step()
        torch.save(predictor, './fg_output.pt')
    predictor.eval()

    print("실제 test")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            output = f_resnet.forward(batch_data.to(device))
            output = predictor.forward(output)
            # argmax = 가장 큰 값의 index를 가져옴
            predicted = torch.argmax(output, 1)
            total += batch_label.size(0)
            correct += (predicted == batch_label.to(device)).sum().item()
    print('총 개수 : %i \n 맞춘 개수 : %i \n 정확도: %d \n 찍었을 때의 정확도 : %d' % (total, correct, 100.0 * correct / total, 10))
