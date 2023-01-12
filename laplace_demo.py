
import cv2 # 我只用它来做图像读写和绘图，没调用它的其它函数哦
import numpy as np # 进行数值计算.
#np.set_printoptions(threshold=np.inf)

# Prewitt 滤波函数
def laplacian(img):

	# 获取图像尺寸
	H, W = img.shape

	# 滤波器系数
	K = np.array([[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]])
	print(type(img))
	re = np.zeros_like(img)
	print(type(re[1,1]))
	img = img.astype("float")
	for i in range(1, img.shape[0] - 1):
		for j in range(1, img.shape[1] - 1):
			if(i==1 and j==1):
				print(img[i-1 : i+2, j-1 : j+2])
				print(K)
				print((img[i-1 : i+2, j-1 : j+2] * K).sum())
			re[i, j] = (img[i-1 : i+2, j-1 : j+2] * K).sum()
			if(i==1 and j==1):
				print(re[i,j])
	print("re:")
	print(re)
	out = re[1:-1, 1:-1]
	#out = np.clip(out, 0, 255)
	return out


# 读取图片
path = 'C:/Users/17865/Desktop/laplace/input/'


file_in = path + 'input1.png' 
file_out = path + 'laplacian_filter.jpg' 
img = cv2.imread(file_in)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)
print(img)
# 调用函数进行sobel滤波
out = laplacian(img)


# 保存图片
cv2.imwrite("laplacian_filter.jpg",out)
cv2.imwrite(file_out, out)
#cv2.imshow("result", out)

cv2.waitKey(0)
#cv2.destroyAllWindows()

print(out)
H, W = out.shape
ll = 0
for i in range(H):
	for j in range(W):
		#print(out[i,j])
		ll = ll+ out[i,j]
print(ll)
