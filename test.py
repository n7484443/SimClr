from torch import Tensor
import torch.utils.data as tu_data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def forward(x, size, tmp):
    x = nn.functional.normalize(x, dim=1)
    # print(x)
    # similarity 계산
    x = torch.mm(x, torch.transpose(x, 0, 1)) / tmp
    # print(x)
    # 대각 부분은 제외되어야 함
    # 즉 대각 부분만 추출 -> 다시 대각 행렬로 변환하여 마스킹 가능
    x = x - torch.diag(torch.diag(x, 0))
    # print(x)
    # down[0] -> up[0] -> down[2] -> up[2] -> 형식
    # 즉 대각 성분을 2칸씩 건너뛰며 추출
    mask = torch.tensor([(i + 1) % 2 for i in range(size - 1)], dtype=torch.float)
    up_mask = torch.diag(mask, 1)
    down_mask = torch.diag(mask, -1)
    mask = up_mask + down_mask
    # print(mask)

    masked_x = x * mask
    masked_x = torch.sum(masked_x, dim=1)
    masked_x = torch.exp(masked_x)
    # print(masked_x)
    e_x = torch.exp(x)
    e_x = e_x - torch.eye(size)
    # print(e_x)
    e_x = torch.sum(e_x, dim=1)
    # print(e_x)
    output = torch.div(masked_x, e_x)
    # print(output_resnet18)
    output = -torch.log(output)
    # print(output_resnet18)
    return torch.sum(output)

if __name__ == '__main__':
    tensor_input = Tensor([[0.8, 0.6], [0.707, 0.707], [0.0, 1], [0.6, 0.8]])
    tensor_output = forward(tensor_input, 4, 1)
    print(tensor_output)