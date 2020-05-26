'''
@ Project Name: Digital Image Processing

@ Author: Honbey, honbey@honbey.com
    Created On 2020-05-10
    Copyright (c) 2020. All rights reserved.

@ Date: 2019-05-10
@ Desc: Image Processing functions
'''

import numpy as np
import matplotlib.pyplot as plt


# 计算两幅单通道图像的协方差矩阵
def covariance(x, y, ddof=0):
    X = np.array(x, dtype=np.float64)
    Y = np.array(y, dtype=np.float64)
    
    avgX = X.mean(axis = 0)
    avgY = Y.mean(axis = 0)
    
    X -= avgX
    Y -= avgY
    cov = np.dot(X.T, Y)
    cov *= np.true_divide(1, X.shape[0] - ddof) # divide N - ddof
    
    return cov

# 计算两幅单通道图像的相关系数矩阵，相当于 MatLab 的 corr() 函数，计算的是列相关性
def corr(x, y, ddof=0):    
    X = np.array(x, dtype=np.float64)
    Y = np.array(y, dtype=np.float64)
    
    cov = covariance(X, Y, ddof=ddof)

    stdX = X.std(axis = 0).reshape(X.shape[1], 1)
    stdY = Y.std(axis = 0).reshape(1, Y.shape[1])
    stdM = np.dot(stdX, stdY)
    
    return cov / stdM

# 相当于 MatLab 的 corrcoef() 函数，先把矩阵转为向量再计算相关性
def corrcoef(x, y):
    X = x.flatten()
    Y = y.flatten()
    return np.corrcoef(X, Y)

# 再量化，可随意设置步长，但是输入图像必须是单通道图像，虽然没有写判别代码
def requalification(image, step):
    size = 256 // step + 1  # 加一因为是linspace函数的特性决定的
                            # 相当于把[1,2]切成两份，但有3个挡板，就像这样 |1|2|
    eList = np.linspace(0, 256, size)
    grayImg = image.copy() # 这里没有判别单通道，调用时注意
    # 双重循环，效率有点低，暂时没想出好对策
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            newValue = binSearch(grayImg[i, j], eList, 0, size - 1)
            grayImg[i, j] = newValue * step + step // 2 # 取每个量化区间的中值，例如[0,3]区间取2
    return grayImg

# 二分查找，提高效率，返回距离最近的索引
def binSearch(e, array, lo, hi):
    while lo < hi:
        mi = (lo + hi) >> 1 # 除以2
        if e < array[mi]:
            hi = mi
        else: lo = mi + 1
    return lo - 1

# 傅里叶变换后选取感兴趣区域逆变换
def ifftROI(fImage, size):
    hSize = fImage.shape[0] // 2
    vSize = fImage.shape[1] // 2
    fsImg = fImage.copy()
    fsImg[hSize-size:hSize+size, vSize-size:vSize+size] = 0
    #fsImg = np.fft.ifftshift(fsImg)
    fftShow = 20 * np.log(np.abs(fsImg) + 1) # 加一防止log(0)错误
    result = np.fft.ifft2(fsImg)
    result = np.abs(result)
    plt.subplot(121), plt.imshow(fftShow, cmap="gray"), plt.title("频谱图"), plt.axis("off")
    plt.subplot(122), plt.imshow(result, cmap="gray"), plt.title("ROI区域置0的傅里叶逆变换"), plt.axis("off")
    plt.show()
    return result

# 均方差和峰值信噪比
def calcMSEPSNR(image1, image2):
    M = image1 - image2
    MSE = np.sum(M.flatten() * M.flatten()) / np.size(image1)    
    #SNR = 10 * np.log10(np.sum(image1.flatten() * image1.flatten()) / MSE / np.size(image2))
    if MSE == 0:
        PSNR = "inf";
    else:
        PSNR = 10 * np.log10((255 ** 2) / MSE)
    print("均方差：", MSE)
    print("峰值信噪比：", PSNR)

# 线性灰度变换
def linearT(image, ltype="c", start=None, end=None, value=None, a=None):
    if image.ndim > 2:
        grayImg = np.dot(image, [0.299,0.587,0.114])
    else:
        grayImg = image.copy()
    
    # 默认使用圆曲线，图像整体会变亮
    x = np.linspace(0, 255, 256)
    if ltype=="c":
        title = "圆曲线"
        y = np.uint8(np.sqrt(65025 - (x - 255) ** 2))
    
    # 灰度反转
    elif ltype=="rv":
        title = "反转变换"
        y = x.copy()
        y = x[::-1]
    
    # 伽马变换
    elif ltype=="gamma":
        title = "伽马变换"
        if value!=None:
            y = np.uint8(np.power(np.linspace(0, 1, 256), value) * 255)
        else:
            y = np.uint8(np.power(np.linspace(0, 1, 256), 2) * 255)
            
    # 对数变换
    elif ltype=="log":
        title = "对数变换"
        if value=="r":
            y = np.uint8((np.power(np.e, x * np.log(1 + 255) / 255) - 1))
        else:
            y = np.uint8((np.log(1 + x) / np.log(1 + 255)) * 255)
    
    # 对比度拉伸
    elif ltype=="cr":
        title = "对比度拉伸"
        y = x.copy()
        if start!=None and end!=None:
            x1, y1 = start
            x2, y2 = end
            if x1 == 0:
                x1 = 1
            elif x2 == 255:
                x2 = 254
        else:
            x1, y1 = (64, 32)
            x2, y2 = (192, 224)
            
        # 计算三条直线的参数
        p1 = (x1, y1)
        p2 = (x2, y2)
        
        k = (p1[1] - 0) / (p1[0] - 0)
        y[0:x1] = k * x[0:x1]
        
        if (p2[0] - p1[0])!=0:
            k = (p2[1] - p1[1]) / (p2[0] - p1[0])
            C = p1[1] - p1[0] * k
            y[x1:x2] = k * x[x1:x2] + C
        
        k = (p2[1] - 255) / (p2[0] - 255)
        C = p2[1] - p2[0] * k
        y[x2:256] = k * x[x2:256] + C
    
    # 基于直线 y=x 自定义变换区间
    elif ltype=="yx":
        title = "灰度级分层(y=x)"
        y = x.copy()
        if value!=None:
            y[start:end] = value
        else:
            y[start:end] = (start + end) // 2
                 
    # 基于直线 y=a 自定义变换区间
    elif start!=None and end!=None and value!=None and a!=None:
        title = "灰度级分层(y=a)"
        y = np.zeros(x.shape, dtype=np.uint8) + a
        y[start:end] = value
    
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            grayImg[i, j] = y[grayImg[i, j]]
    plt.subplot(131), plt.plot(x, y), plt.title(title, y=0.72), plt.axis("equal"), plt.axis("off")
    plt.subplot(132), plt.imshow(image, cmap="gray"), plt.title("原图"), plt.axis("off")
    plt.subplot(133), plt.imshow(grayImg, cmap="gray"), plt.title("结果"), plt.axis("off")
    plt.show()
    return grayImg

# 直方图
def toHist(image):    
    L = np.linspace(0, 0, 256, dtype=np.uint32)
    # 统计频数
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            L[image[i, j]] += 1
    return L

# 直方图均衡化
def histNorm(image):
    if image.ndim > 2:
        grayImg = np.dot(image, [0.299,0.587,0.114])
    else:
        grayImg = image.copy()
    
    L = toHist(grayImg) # 先得到原图的直方图
        
    P = L / grayImg.size
    S = np.linspace(0, 0, 256)
    
    # 计算Sr
    for i in range(256):
        S[i] = 255 * np.sum(P[0:i]);
        if S[i] - np.trunc(S[i]) >= 0.5:
            S[i] = np.uint8(np.ceil(S[i]))
        else:
            S[i] = np.uint8(np.floor(S[i]))
    
    # 图像均衡化
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            grayImg[i, j] = S[grayImg[i, j]]
    return grayImg

# 高斯噪声和椒盐噪声函数
def gasussNoise(image, mu=0, sigma=0.001):
    noise = np.random.normal(mu, sigma ** 0.5, image.shape)
    # 加性噪声通过相加实现
    result = np.array(image / 255, dtype=np.float64) + noise
    result = np.clip(result, 0., 1.0)
    result = np.uint8(result * 255)
    return result
    
def pepperNoise(image, SNR=0.05):
    result = image.copy()
    cnt = np.int32(image.size * SNR)
    for i in range(cnt):
        # 随机选取位置，随机置0或置255
        randX = np.random.randint(0, image.shape[0], 1)[0]
        randY = np.random.randint(0, image.shape[0], 1)[0]
        randV = np.random.randint(0, 2, 1)[0]
        if randV:
            result[randX][randY] = 255
        else:
            result[randX][randY] = 0
    return result

# 均值滤波
def average(data):
    mean = sum(data) // len(data)
    return mean

# 中值滤波
def median(data):
    data.sort()
    half = len(data) // 2
    return data[half]

# 滤波器
def filters(image, model="A", size=3):
    s = size // 2 # 填充范围
    # 输出图像
    result = image.copy()
    # 防止无意中改变图像像素值而导致偏差
    padImg = np.pad(image, (s, s), mode="edge")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 使用分片代替循环，提高效率
            data = padImg[i:i+size, j:j+size]
            # 均匀滤波
            if model=="A":
                result[i][j] = np.round(data.mean())
                
            # 中值滤波
            elif model=="M":
                # np.median函数不知道为啥效率没有直接使用索引高，按理说numpy内部实现也应该是基于索引
                #result[i][j] = np.median(np.sort(data, axis=None))
                result[i][j] = np.sort(data, axis=None)[size**2 // 2]
            
            elif model=="AG":
                filter1 = np.array([
                            [1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1],
                        ])
                R = (filter1 * data).sum() // 16
                result[i][j] = R
            
            # 拉普拉斯算子
            elif model=="L1":
                filter1 = np.array([
                            [-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1],
                        ])
                R = (filter1 * data).sum()
                result[i][j] += R
            elif model=="L2":
                filter1 = np.array([
                            [0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0],
                        ])
                R = (filter1 * data).sum()
                result[i][j] += R
            
            # Sobel算子
            elif model=="S":
                filter1 = np.array([
                            [-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1],
                        ])
                filter2 = np.array([
                            [-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1],
                        ])
                X = (filter1 * data).sum()
                Y = (filter2 * data).sum()
                result[i][j] = np.sqrt(X**2 + Y**2)
                        
    return np.clip(result, 0, 255)

def grayImageJudge(src_image, method="w"):
    if src_image.ndim > 2:
        if method == "w":
            dst_image = np.dot(src_image, [0.299,0.587,0.114])
    else:
        dst_image = src_image.copy()
    return dst_image

def zoomByInterpolation(src_image, xratio=0.5, yratio=0.5, method="neighbor"):
    image = grayImageJudge(src_image)
    dst_image = np.zeros((int(src_image.shape[0]*xratio), int(src_image.shape[1]*yratio)), dtype=np.uint8)
    x_r = 1 / xratio
    y_r = 1 / yratio

    for i in range(dst_image.shape[0]):
        for j in range(dst_image.shape[1]):
            if method == "neighbor" or method == "n":
                src_x = (int)(np.floor(i*x_r))
                src_y = (int)(np.floor(j*y_r))
                dst_image[i][j] = image[src_x][src_y];
            if method == "bilinear" or method == "b":
                src_x = (i+0.5)*x_r-0.5
                src_y = (j+0.5)*y_r-0.5

                src_x1 = (int)(np.floor(src_x))
                src_y1 = (int)(np.floor(src_y))
                src_x2 = min(src_x1+1, image.shape[0]-1)
                src_y2 = min(src_y1+1, image.shape[1]-1)
                
                r1 = (src_x2 - src_x) * image[src_x1][src_y1] + (src_x - src_x1) * image[src_x2][src_y1]
                r2 = (src_x2 - src_x) * image[src_x1][src_y2] + (src_x - src_x1) * image[src_x2][src_y2]
                dst_image[i][j] = (int)((src_y2 - src_y) * r1 + (src_y - src_y1) * r2)
                
    return dst_image;
                
    
def globalThresholdSegmentation(src_image, v=None):
    dst_image = grayImageJudge(src_image)
    t1, t2 = 128, 130
    while np.abs(t2 - t1) > 0.0001 and v==None:
        print(t2,t1)
        t1 = t2
        
        index_zero = (int)(np.min(src_image))
        index_one = (int)(np.floor(t1))
        total_zero = 0
        total_one = 0
        count_zero = 0
        count_one = 0
        
        histgram_list = myipf.toHist(dst_image)
        for index_zero in range(index_one):
            total_zero += histgram_list[index_zero]*index_zero
            count_zero += histgram_list[index_zero]
        for index_one in range(np.max(src_image)):
            total_one += histgram_list[index_one]*index_one
            count_one += histgram_list[index_one]
              
        t2 = (total_one/count_one + total_zero/count_zero) / 2
    
    if v != None:
        t2 = v
    for i in range(src_image.shape[0]):
            for j in range(src_image.shape[1]):
                if src_image[i][j] > t2:
                    dst_image[i][j] = 1;
                else:
                    dst_image[i][j] = 0;
    return dst_image;

def main():
    print("My image processing functions.")

if __name__ == "__main__":
    main()