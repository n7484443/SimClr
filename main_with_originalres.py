import torch.utils.data as tu_data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch_optimizer as optim
from torch import Tensor
from torchinfo import summary
import numpy as np


def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # pin_memory : GPU에 데이터를 미리 전송
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# 이미지 텐서를 색상 변환, 크롭
def distortion(tensor_image: Tensor, strength: float) -> Tensor:
    # [batch_size, rgb, width, height]
    img_size = tensor_image.shape[2]
    color_transform = transforms.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength,
                                             saturation=0.8 * strength, hue=0.2 * strength)
    # 0.5~1 사이의 크기, 0.5~2 사이의 종횡비로 랜덤하게 크롭
    crop_transform = transforms.RandomResizedCrop((img_size, img_size), scale=(0.6, 1), ratio=(0.5, 2.0), )
    flip_horizon = transforms.RandomHorizontalFlip(p=0.5)
    flip_vertical = transforms.RandomVerticalFlip(p=0.5)
    transform_several = torch.nn.Sequential(
        crop_transform,
        flip_vertical,
        flip_horizon,
        color_transform,
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
        x = self.relu(x)
        x = self.hidden(x)
        return x


# 출처 : https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
class SimCLR_Loss(nn.Module):
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


if __name__ == '__main__':
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 1024
    hyper_batch_size_predictor = 32
    hyper_epoch = 60
    hyper_epoch_predictor = 60
    lr = 0.001
    momentum = 0.9
    lr_predictor = 0.0003
    temperature = 0.05
    strength = 0.8

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    # rgb 3개,
    size = 32
    output_size = 10
    f_resnet = torchvision.models.resnet18(weights=None)  # torchvision.models.ResNet18_Weights
    num_input = f_resnet.fc.in_features
    f_resnet.fc = nn.Linear(num_input, size)

    g_small = SimpleMLP(input_size=size, output_size=size)

    fg = nn.Sequential(f_resnet, g_small)
    fg.to(device)
    fg.train()

    loss_function = SimCLR_Loss(temperature=temperature)

    summary(fg, input_size=(hyper_batch_size, 3, 32, 32))

    optimizer = optim.LARS(fg.parameters(), lr=lr, momentum=momentum)
    # train
    data_output = []
    for epoch in range(hyper_epoch):
        first = False
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_train = 0
        for batch_data, batch_label in trainLoader:
            batch_size = batch_data.shape[0]
            batch_data_cuda = batch_data.to(device)
            batch_data_distorted = distortion(tensor_image=batch_data_cuda, strength=strength)
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

            loss = loss_function.forward(batch_data_after, batch_distorted_after, batch_size)
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()

        loss_sum_test = 0
        with torch.no_grad():
            for batch_data, batch_label in testLoader:
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_distorted = distortion(tensor_image=batch_data_cuda, strength=strength)
                batch_data_after = fg.forward(batch_data_cuda)
                batch_distorted_after = fg.forward(batch_data_distorted)
                loss = loss_function.forward(batch_data_after, batch_distorted_after, batch_size)
                loss_sum_test += loss.item()

        loss_sum_train /= len(trainLoader)
        loss_sum_test /= len(testLoader)
        print('Epoch : %d, Avg Loss : %.4f, Validation Loss : %.4f' % (
            epoch, loss_sum_train, loss_sum_test))
        data_output.append((loss_sum_train, loss_sum_test))

    plt.subplot(2, 1, 1)
    range_x = np.arange(0, hyper_epoch, 1)
    plt.plot(range_x, [x[0] for x in data_output], label='Training loss', color='red')
    plt.plot(range_x, [x[1] for x in data_output], label='Validation loss', color='blue')
    plt.legend()
    plt.title(f'lr: {lr_predictor} batch:{hyper_batch_size} epoch: {hyper_epoch} temp:{temperature}')

    print("FG 학습 완료. 이제 F의 output을 실제 dataset의 label과 연결.")

    trainLoader, testLoader = load_image(batch_size=hyper_batch_size_predictor)
    # predictor
    predictor = nn.Linear(size, output_size)
    predictor.to(device)
    # fg output 과 실제 dataset label의 연결
    simple_loss_function = nn.CrossEntropyLoss()
    simple_loss_function.to(device)

    f_resnet.eval()
    predictor.train()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr_predictor)
    data_output = []
    for epoch in range(hyper_epoch_predictor):
        # batch_size * 2, rgb, x, y 의 데이터 형태
        loss_sum_test = 0
        loss_sum_train = 0
        for batch_data, batch_label in trainLoader:
            batch_size = batch_data.shape[0]

            with torch.no_grad():
                output = f_resnet.forward(batch_data.to(device))
            expect = predictor.forward(output)
            loss = simple_loss_function(expect, batch_label.to(device))
            loss_sum_train += loss.item()

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()
            optimizer.step()
            # print(loss)
        with torch.no_grad():
            for batch_data, batch_label in testLoader:
                batch_size = batch_data.shape[0]
                batch_data_cuda = batch_data.to(device)
                batch_data_after = fg.forward(batch_data_cuda)
                expect = predictor.forward(batch_data_after)
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
    plt.legend()
    plt.title(f'predictor \nlr: {lr} batch:{hyper_batch_size_predictor} epoch: {hyper_epoch_predictor}')

    plt.tight_layout()
    plt.show()

    predictor.eval()

    print("실제 test")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            output = fg.forward(batch_data.to(device))
            output = predictor.forward(output)
            # argmax = 가장 큰 값의 index를 가져옴
            predicted = torch.argmax(output, 1)
            total += batch_label.size(0)
            correct += (predicted == batch_label.to(device)).sum().item()
    print(f'총 개수 : {total} \n 맞춘 개수 : {correct} \n 정확도: {100.0 * correct / total} \n 찍었을 때의 정확도 : {10}')
