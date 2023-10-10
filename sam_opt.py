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

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


def load_image(batch_size) -> (tu_data.DataLoader, tu_data.DataLoader):
    # X => (X - mean)/standard_deviations (정규분포의 Normalization)
    # 이미지는 일반적으로 0~255 혹은 0~1, 여기선 0~1 의 값
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test data와 train data는 분리되어야 함. 미 분리시 test data가 train data에 영향을 줄 수 있음
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # pin_memory : GPU에 데이터를 미리 전송
    train_loader = tu_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = tu_data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader


if __name__ == '__main__':
    want_train = True
    device = torch.device("cuda")
    hyper_batch_size = 256
    hyper_epoch = 30

    testLoader: tu_data.DataLoader
    trainLoader: tu_data.DataLoader
    trainLoader, testLoader = load_image(batch_size=hyper_batch_size)

    resnet_0 = torchvision.models.resnet18(pretrained=True).to(device)
    resnet_1 = torchvision.models.resnet18(pretrained=True).to(device)
    simple_loss = nn.CrossEntropyLoss().to(device)

    sam_opt = SAM(base_optimizer=torch.optim.SGD, params=resnet_0.parameters(), lr=0.1)
    sgd_opt = torch.optim.SGD(params=resnet_1.parameters(), lr=0.1)

    for epoch in range(hyper_epoch):
        for batch_data, batch_label in trainLoader:
            batch_data_cuda = batch_data.to(device)
            batch_label_cuda = batch_label.to(device)
            sam_output = resnet_0.forward(batch_data_cuda)
            sgd_output = resnet_1.forward(batch_data_cuda)
            loss_sam_output = simple_loss(sam_output, batch_label_cuda)
            loss_sgd_output = simple_loss(sgd_output, batch_label_cuda)

            sam_opt.zero_grad()  # gradient 초기화
            sgd_opt.zero_grad()  # gradient 초기화
            loss_sam_output.backward()
            loss_sgd_output.backward()

            sam_opt.first_step(zero_grad=True)
            sam_output = resnet_0.forward(batch_data_cuda)
            simple_loss(sam_output, batch_label_cuda).backward()
            sam_opt.second_step(zero_grad=True)
            sgd_opt.step()
        print(f"epoch : {epoch}")

    correct_0 = 0
    correct_1 = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in testLoader:
            batch_data_cuda = batch_data.to(device)
            batch_label_cuda = batch_label.to(device)
            output_0 = resnet_0.forward(batch_data_cuda)
            output_1 = resnet_1.forward(batch_data_cuda)
            predicted_0 = torch.argmax(output_0, 1)
            predicted_1 = torch.argmax(output_1, 1)
            total += batch_label_cuda.size(0)
            correct_0 += (predicted_0 == batch_label_cuda).sum().item()
            correct_1 += (predicted_1 == batch_label_cuda).sum().item()

    print(f"sam: {100 * correct_0 / total} sgd: {100 * correct_1 / total}")
