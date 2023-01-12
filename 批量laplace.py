import os          #导入os模块主要用于文件的读写
import argparse    #导入argpase主要是用来命令行运行时参数的配置
import cv2         #图像处理模块
import numpy as np
import torch
 
#parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')    #创建一个参数解析对象，为解析对象添加描述语句，而这个描述语句是当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，会打印这些描述信息
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images", action="store_true")    #为函数添加参数k和参数keepdims，并且设置action="store_true"，也就是当命令行提及到这两个参数的时候，参数设置为true，如果没提及那就是默认值（如果用了default制定了默认值的话）
parser.add_argument('--color_img_dir', type=str, default=r'C:\Users\17865\Desktop\Laplace_Filter\input',        #设置高分辨率图片路径参数
                    help='path to Color image dir')
parser.add_argument('--laplace_img_dir', type=str, default=r'C:\Users\17865\Desktop\Laplace_Filter\result',       #设置低分辨率路径参数
                    help='path to desired output dir for grayed images')
args = parser.parse_args()                                #调用parse_args()方法对参数进行解析；解析成功之后即可使用
 
color_image_dir = args.color_img_dir              #从参数列表中取出高分辨率图像路径
laplace_image_dir = args.gray_img_dir              #从参数列表中取出低分辨率图像路径
 
print(args.color_img_dir)                      #将热红外热图像路径打印出来
print(args.laplace_img_dir)                      #将热红外灰度图像路径打印出来
 
 
#create LR image dirs
#在低分辨率图像路径中创建每个下采样倍率的文件夹
os.makedirs(laplace_image_dir, exist_ok=True)       #创建保存灰度图的文件夹，exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常       

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")      #在这里用一个元组保存支持进行下采样的图像的后缀格式

num = 0 #初始化一个拉普拉斯滤波计数器
#laplaceing HR images
#对热图像进行拉普拉斯滤波
for filename in os.listdir(color_image_dir):             #遍历热图像文件夹中的每一个文件
    if not filename.endswith(supported_img_formats):    #如果文件的后缀名不是支持灰度化的图片格式，那么就跳过这张图片
        continue
 
    name, ext = os.path.splitext(filename)              #os.path.splitext(“文件路径”)：分离文件名与扩展名；默认返回(fname,fextension)元组
    #在这里，我们将遍历的每个文件的文件名存在变量name中，后缀存在ext中
 
    #Read HR image
    color_img = cv2.imread(os.path.join(color_image_dir, filename))  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    #图像灰度化
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor=img_tensor.float()
    conv= torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=0, bias=False)  #创造一个拉普拉斯算子
    conv.weight.data = torch.Tensor([[[[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]]])
    conv.weight.data.requires_grad = False       #冻结拉普拉斯算子的权重
    img_tensor_conv = conv(img_tensor)
    out_conv = img_tensor_conv.detach.numpy()
    """
    os.path.join()函数用于路径拼接文件路径,在这里也就是将文件所在目录和文件名拼接在一起，获得文件完整的路径
    cv2.imread:为 opencv-python 包的读取图片的函数,cv2.imread()有两个参数,第一个参数filename是图片路径,第二个参数flag表示图片读取模式,共有三种
    cv2.IMREAD_COLOR:加载彩色图片,这个是默认参数,可以直接写1。
    cv2.IMREAD_GRAYSCALE:以灰度模式加载图片,可以直接写0。
    cv2.IMREAD_UNCHANGED:包括alpha(包括透明度通道)，可以直接写-1
    cv2.imread()读取图片后以多维数组的形式保存图片信息，前两维表示图片的像素坐标,最后一维表示图片的通道索引,具体图像的通道数由图片的格式来决定
    cv2.imread()读取的图片是bgr格式而不是rgb格式
    """   

 
    cv2.imwrite(os.path.join(laplace_image_dir, filename.split('.')[0]+ext), out_conv)  #将文件写入准备好的灰度图片文件夹中
    num = num+1
    print(num)

 
print("已经所有图片进行拉普拉斯滤波")