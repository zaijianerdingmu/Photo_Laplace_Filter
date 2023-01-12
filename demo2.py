
import cv2 # 我只用它来做图像读写和绘图，没调用它的其它函数哦
import numpy as np # 进行数值计算
import torch
import torchvision
import math



# 读取图片
path = 'C:/Users/17865/Desktop/laplace/input/'


file_in = path + 'input1.png' 
file_out = path + 'laplacian_filter.jpg' 
img = cv2.imread(file_in)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_tensor = torch.tensor(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.unsqueeze(0)
img_tensor=img_tensor.float()
#print(img_tensor)
# 调用函数进行sobel滤波
img_tensor = torch.tensor([[[[116., 121., 124.],
                              [118.,122.,124.],
                              [119.,122.,124.]
                              ]]])
print(img_tensor)
print(img_tensor.dtype)
conv= torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=0, bias=False)  #创造一个拉普拉斯算子
conv.weight.data = torch.Tensor([[[[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]]])
conv.weight.data.requires_grad = False       #冻结拉普拉斯算子的权重
img_tensor_conv = conv(img_tensor)
#print(img_tensor_conv)
out_conv = img_tensor_conv.detach.numpy()


# 保存图片
cv2.imwrite(file_out, out_conv)

print(out_conv.shape)
H, W = out_conv.shape
ll = 0
for i in range(H):
	for j in range(W):
		print(out_conv[i,j])
        #print(out_conv[i,j])
		ll = ll+ out_conv[i,j]
print(ll)



