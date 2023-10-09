```python
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
```
import문 정리.

```python
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
```
이미지 다운로드, 배치로 분리, 학습 데이터와 테스트 데이터를 분리하여 로드.
```python
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
```
이미지를 랜덤하게 변환하여 리턴.
크롭, 뒤집기, 색상 변환을 랜덤하게 한 후 배치 정규화.
[batch_size, rgb(3), width(32), height(32)] 크기의 텐서를 반환
```python
# identity resnet 구현
class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        if in_channel != out_channel:
            self.sequentialLayer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, bias=False, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(out_channel)
            )
            # 1x1 conv, stride = 2(pooling을 안 쓰는 건 일종의 철학), 크기 조절을 위한 projection mapping
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.sequentialLayer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, bias=False, padding='same'),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, bias=False, padding='same'),
                nn.BatchNorm2d(out_channel)
            )
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sequentialLayer(x) + self.shortcut(x)
        x = self.relu(x)
        return x
```
본 코드에서는 resnet을 직접 구현한 버전(main.py)와 resnet-34를 사용한 버전(main_with_originalres.py)가 존재함.
여기서는 resnet을 직접 구현한 버전.


```python
# conv output size = ceil((i + 2p - k) / s) + 1
class Resnet(nn.Module):
    def __init__(self, size):
        super().__init__()
        # rgb 3개로 16개의 채널 만듬
        # 16 * image width(32) * image height(32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # 16 * 32 * 32 -> 32 * 16 * 16
        self.conv2 = nn.Sequential(
            ResnetBlock(16, 32),
            *[ResnetBlock(32, 32) for _ in range(4)]
        )
        # 32 * 16 * 16 -> 64 * 8 * 8
        self.conv3 = nn.Sequential(
            ResnetBlock(32, 64),
            *[ResnetBlock(64, 64) for _ in range(4)]
        )
        # 64 * 8 * 8 -> 128 * 4 * 4
        self.conv4 = nn.Sequential(
            ResnetBlock(64, 128),
            *[ResnetBlock(128, 128) for _ in range(4)]
        )
        # 전체 평균, 즉 128 * 1 * 1
        # AvgPool2d 는 Conv2d와 같이 커널을 이동시키며 평균을 구함
        # AdaptiveAvgPool2d 는 AvgPool2d와 비슷하나 특정 크기로 자동으로 맞춰줌
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten으로 128 * 1 * 1을 128 벡터로 바꾼 후 fully connected layer로 128을 size 크기로 변환
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x
```
Resnet Block을 사용한 Resnet 구현.
conv -> batchnorm -> relu -> resnet -> resnet -> ... -> flatten -> full connected layer
```python
class SimpleMLP(nn.Module):
    # SimCLR에 사용된 최소화된 MLP
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        return x
```
projection layer로 사용된 MLP 구현.
비교적 간단한 구현체로, non-linear하게 하기 위해 relu를 사용, hidden layer을 사용하였다.

```python
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
```
위는 SimCLR의 Loss function을 구현한 것으로, 논문의 코드를 for문을 이용하여 구현하면 속도가 많이 느리다.
따라서 행렬 연산을 이용하여 구현하였다.

|   | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | a | c | d | f |
| 2 | b | d | e | g |
의 normalized 된 feature 텐서가 있으면 이의 Cosine 유사도 행렬은

|   | 1       | 2       | 3       | 4       |
|---|---------|---------|---------|---------|
| 1 | 1       | ac + bd | ad + be | af + bg |
| 2 | ac + bd | 1       | cd + de | cf + dg |
| 3 | ad + be | cd + de | 1       | df + eg |
| 4 | af + bg | cf + dg | df + eg | 1       |
의 행렬이 된다.
여기서 대각 성분을 torch.eye 의 Idenetity 행렬을 만들어서 빼 주면

|   | 1       | 2       | 3       | 4       |
|---|---------|---------|---------|---------|
| 1 | 0       | ac + bd | ad + be | af + bg |
| 2 | ac + bd | 0       | cd + de | cf + dg |
| 3 | ad + be | cd + de | 0       | df + eg |
| 4 | af + bg | cf + dg | df + eg | 0       |
가 나온다.
이를 torch.exp로 지수화하면

|   | 1                                              | 2                                              | 3                                              | 4                                              |
|---|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| 1 | 1                                              | <span style="background-color:red">e^(ac + bd) | e^(ad + be)                                    | e^(af + bg)                                    |
| 2 | <span style="background-color:red">e^(ac + bd) | 1                                              | e^(cd + de)                                    | e^(cf + dg)                                    |
| 3 | e^(ad + be)                                    | e^(cd + de)                                    | 1                                              | <span style="background-color:red">e^(df + eg) |
| 4 | e^(af + bg)                                    | e^(cf + dg)                                    | <span style="background-color:red">e^(df + eg) | 1                                              |
이며, 이 때 l 벡터는 색칠된 부분을 각 column 의 합 - 1로 나눠주는 것과 같다.
여기서 분자는 1 0 1 0 1 0 ... 의 벡터를 torch.diag로 한칸 위, 한칸 아래로 대각화 하여 만든 mask 행렬과의 곱으로 구할 수 있다.
분모는 Idenetity 행렬을 뺀 후 각 column의 합을 구하면 된다.

이를 log를 취해 주면 값이 나온다.


```python
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
```
이 함수는 다른 사람이 구한 방법으로, 거의 비슷한 방식이다.
다만 직접 구현한 방식은 짝수에 augmentation image의 특성벡터를, 홀수에는 distorted image의 특성벡터를 넣어서 계산하였다.
이 방식은 torch.cat으로 augmentation image의 특성 벡터를 앞에, distorted image의 특성 벡터를 뒤에 넣어서 계산하였다.

```python
if __name__ == '__main__':
    writer = SummaryWriter()
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 50
    hyper_epoch = 50

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)
    # rgb 3개,
    size = 32
    output_size = 10
    f_resnet = Resnet(size=size)
    loss_function = SimCLR_Loss(batch_size=hyper_batch_size, temperature=0.1)

    summary(f_resnet, input_size=(hyper_batch_size, 3, 32, 32))

    f_resnet.to(device)
```
이미지와 각종 설정을 하는 부분이다.
```python
    if os.path.exists('./model.pt'):
        f_resnet = torch.load('./model.pt')
    if not os.path.exists('./model.pt') or False:
        g_small = SimpleMLP(input_size=size, output_size=output_size)
        g_small.to(device)
        fg = nn.Sequential(
            f_resnet,
            g_small
        )
        fg.train()
        optimizer = torch.optim.Adam(fg.parameters(), lr=0.00001)
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
```
이를 통해 f_resnet과 g_small의 학습을 하였고, g_small을 거치면 이미지의 여러 유용한 특성들이 제거되므로 g_small 뒤에 추정 모델을 달지 않고 
f_resnet 뒤에 달았다.
```python
    # predictor
    predictor = nn.Sequential(
        nn.Linear(size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        # class는 10개
    )
```
predictor 모델은 총 4개의 레이어로 구성되어 있다
```python
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
                with torch.no_grad():
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
```
predictor 모델의 학습을 하는 코드이며, lr scheduler을 사용하여 lr 값을 점점 줄였다.
또한 f_resnet에는 그래디언트가 흐르지 않게 freeze 하였다.
```python
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
```
마지막으로 테스트를 하는 코드이다.
최적화를 위해 그래디언트를 꺼 주었다.


이러한 과정으로 예측모델을 세울 수 있었고. 이에 대한 결과는 output.txt에 기록하였다.
배치 값과 epoch 값에 따라서 결과가 달라지지만, epoch가 약 200회를 넘어서야만 45%의 정확도를 가질 수 있었다.
이에 대한 원인 추측으론 이미지의 크기가 너무 작아서 여러 변형을 하면 원본이 심하게 손상되어 특성을 잃어버리게 되는 것이 하나로 추측되고 또 다른 원인으로는 f_resnet의 구조가 잘못되었을 수도 있다고 생각했다.
따라서 f_resnet을 torchvision의 모델을 사용해봤지만, 결과는 그대로였다.
혹시 loss값 계산이 문제일까 싶어서 다른 사람의 코드를 써 봤지만 그대로였다.

마지막으로 배치 사이즈를 50 정도로 논문에 나온 것보다 매우 작게 해보았더니 loss값이 상당히 낮게 나왔다.
또한 정확도도 향상되었다.