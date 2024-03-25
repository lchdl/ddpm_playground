import argparse
import torch
import torch.nn as nn
import numpy as np
import time

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(1,512,kernel_size=3, stride=1, padding=1))
        for _ in range(10): self.layers.append( nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1) )
        self.layers = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.layers(x)

def f(x):
    numel = torch.numel(x)
    return torch.sum(x)/numel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_index', help='Running model on which gpu?', type=int)
    parser.add_argument('model_size', help='Larger value results in more VRAM consumption.', type=int)
    args = parser.parse_args()
    gpu_index = args.gpu_index
    model_size = args.model_size
    if model_size<4: 
        print('model size should >=4.')
        model_size=4
    # model_size: 16, 757M
    # model_size: 32, 
    model = net().cuda(gpu_index)
    optim = torch.optim.Adam(model.parameters())
    x = torch.zeros(1, 1, model_size, model_size).cuda(gpu_index).float()
    print('Occupying GPU %d...' % gpu_index)
    hotrun=10
    while True:
        y: torch.Tensor = model(x)
        l = f(y)
        optim.zero_grad()
        l.backward(retain_graph=False)
        optim.step()
        if hotrun > 0:
            hotrun-=1
        else:
            time.sleep(10)

