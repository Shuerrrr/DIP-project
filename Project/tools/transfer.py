import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math

## 1. 计算patch与目标块overlap的差值
## 2. 给定目标快坐标在纹理图遍历找到overlap差值与patch差值和最小的块坐标
## 3. overlap最小误差路径
## 4.transfer转移函数
## 5. 遍历patchSize和alpha
## transfer中res的第一个块（path最小误差即可），随后两个都用

def getOverlapError(patch,syn,patchSize,ovSize,x,y):  # 计算与syn扫描位置的误差
    Error = 0
    if x>0:  # 不是最左边的，需要计算与前一个的overlapError
        leftError = np.power((patch[:, :ovSize] - syn[y:y + patchSize, x:x+ovSize]),2)
        Error += np.sum(leftError)
    if y>0:
        upError = np.power((patch[:ovSize, :] - syn[y:y + ovSize, x:x + patchSize]), 2)
        Error += np.sum(upError)
    if x > 0 and y > 0:
        cornerError = np.power((patch[:ovSize, :ovSize] - syn[y:y + ovSize, x:x + ovSize]), 2)
        Error -= np.sum(cornerError)
    return Error


# def minCostPath2(errors):
#     # dynamic programming, unused
#     errors = np.pad(errors, [(0, 0), (1, 1)],
#                     mode='constant',
#                     constant_values=np.inf)
#
#     cumError = errors[0].copy()
#     paths = np.zeros_like(errors, dtype=int)
#
#     for i in range(1, len(errors)):
#         M = cumError
#         L = np.roll(M, 1)
#         R = np.roll(M, -1)
#
#         # optimize with np.choose?
#         cumError = np.min((L, M, R), axis=0) + errors[i]
#         paths[i] = np.argmin((L, M, R), axis=0)
#
#     paths -= 1
#
#     minCutPath = [np.argmin(cumError)]
#     for i in reversed(range(1, len(errors))):
#         minCutPath.append(minCutPath[-1] + paths[i][max(minCutPath[-1],-2)])
#         # print()
#         print('index',minCutPath[-1])
#
#     return map(lambda x: x - 1, reversed(minCutPath))

# def minCostPath(errors):
#     # print('errors:', errors.shape)
#     m, n = errors.shape
#     # 创建累积误差矩阵和路径矩阵
#     cumError = np.zeros((m, n))
#     paths = np.zeros((m, n), dtype=int)
#
#     # 初始化累积误差矩阵的第一行
#     cumError[0] = errors[0]
#
#     # 填充累积误差矩阵和路径矩阵
#     for i in range(1, m):
#         for j in range(n):
#             # 考虑从上一行的左、中、右三个位置移动到当前元素的代价
#             left = cumError[i - 1, j - 1] if j > 0 else np.inf
#             middle = cumError[i - 1, j]
#             right = cumError[i - 1, j + 1] if j < n - 1 else np.inf
#
#             # 找到最小的代价并记录路径
#             min_cost = min(left, middle, right)
#             cumError[i, j] = errors[i, j] + min_cost
#             if min_cost == left:
#                 paths[i, j] = j - 1
#             elif min_cost == middle:
#                 paths[i, j] = j   #记录最小值对应坐标
#             else:
#                 paths[i, j] = j + 1
#     # 回溯找到最小路径
#     minCutPath = []
#     # print('shape, 所有累积误差：', cumError.shape, cumError[-1])
#     j = np.argmin(cumError[-1]) #找到最后一行的最小值
#     minCutPath.append(j)
#
#     for i in range(m - 1, 0, -1):#反向遍历
#         j = paths[i, j]
#         minCutPath.append(j)
#     # 由于我们是从最后一行回溯，所以需要反转路径
#     minCutPath.reverse()
#
#     return minCutPath
def minCostPath(errors):
    m, n = errors.shape

    # 创建累积误差矩阵和路径矩阵
    cumError = np.zeros((m, n))
    paths = np.zeros((m, n), dtype=int)

    # 初始化累积误差矩阵的第一行
    cumError[0] = errors[0]

    # 填充累积误差矩阵和路径矩阵
    for i in range(1, m):
        left_shifted = np.roll(cumError[i - 1], 1)
        right_shifted = np.roll(cumError[i - 1], -1)

        left_shifted[0] = np.inf
        right_shifted[-1] = np.inf

        min_costs = np.minimum(np.minimum(left_shifted, cumError[i - 1]), right_shifted)
        cumError[i] = errors[i] + min_costs

        paths[i] = np.argmin([left_shifted, cumError[i - 1], right_shifted], axis=0) - 1

    # 回溯找到最小路径
    minCutPath = []
    j = np.argmin(cumError[-1])  # 找到最后一行的最小值
    minCutPath.append(j)

    for i in range(m - 1, 0, -1):  # 反向遍历
        j += paths[i, j]
        minCutPath.append(j)

    # 由于我们是从最后一行回溯，所以需要反转路径
    minCutPath.reverse()

    return minCutPath


def minCostPatch(patch, patchSize,overlap, res, y, x):
    patch = patch.copy()
    dy, dx = patch.shape[0:2]
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        # print('left:', left.shape)
        leftL2 = np.sum(left**2, axis=2) #注意通道数
        for i, j in enumerate(minCostPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        # print('upL2:', upL2.shape)

        for j, i in enumerate(minCostPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch

def computeEdgeGradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gradX**2 + gradY**2)
    return gradient

def luminanceRemap(texture, target):
    texture_yuv = cv2.cvtColor(texture, cv2.COLOR_BGR2YCrCb)
    target_yuv = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    
    texture_y = texture_yuv[:, :, 0]
    target_y = target_yuv[:, :, 0]
    
    mean_tex = np.mean(texture_y)
    std_tex = np.std(texture_y)
    mean_tgt = np.mean(target_y)
    std_tgt = np.std(target_y)
    
    remapped_texture_y = ((texture_y - mean_tex) * (std_tgt / std_tex)) + mean_tgt
    remapped_texture_y = np.clip(remapped_texture_y, 0, 255).astype(np.uint8)
    
    texture_yuv[:, :, 0] = remapped_texture_y
    remapped_texture = cv2.cvtColor(texture_yuv, cv2.COLOR_YCrCb2BGR)
    
    return remapped_texture

def getBestPatch(texture, textureMap, targetMap, syn,x,y,patchSize,ovSize,alpha):   # ovSize通常为块size的1/6 , syn为正在生成的图像
    H, W = textureMap.shape[:2]
    Errors = np.zeros((H - patchSize, W - patchSize))   # 存储误差的矩阵
    targetMapPatch = targetMap[y:y + patchSize, x:x + patchSize]   # 从目标图区当前扫描到的块
    row, col = targetMapPatch.shape[:2]  # 重点!防止到最后截断patch，两个patch不匹配


    for i in range(H - patchSize):
        for j in range(W - patchSize):  # 遍历Map前后的texture图中所有的块，用第一个像素代替它与目标矩阵的误差
            srcPatch = texture[i:i + patchSize, j:j + patchSize] # 原图块
            overlapError = getOverlapError(srcPatch, syn, patchSize, ovSize, x, y)  # 原图与syn的重叠误差

            textureMapPatch = textureMap[i:i + row, j:j + col]   # 用于判断：计算patch误差,且保证和textureMap维度一致

            # print('Size for two patch:', targetMapPatch.shape, textureMapPatch.shape)
            patchError = np.sum(np.power(targetMapPatch - textureMapPatch, 2))

            Errors[i, j] = alpha * overlapError + (1 - alpha) * patchError

    # print('Another patch error:', Errors[:3])

    min_index = np.unravel_index(np.argmin(Errors), Errors.shape)
    print('Another patch index:', min_index)

    a, b = min_index[:2]

    return texture[a:a+row, b:b+col]

def getBestPatch_enhanced(texture, textureMap, targetMap, syn, x, y, patchSize, ovSize, alpha):
    H, W = textureMap.shape[:2]
    Errors = np.zeros((H - patchSize, W - patchSize))   
    targetMapPatch = targetMap[y:y + patchSize, x:x + patchSize] 
    row, col = targetMapPatch.shape[:2]

    edge_gradient = computeEdgeGradient(targetMapPatch)
    texture = luminanceRemap(texture, targetMapPatch)

    for i in range(H - patchSize):
        for j in range(W - patchSize):  
            srcPatch = texture[i:i + patchSize, j:j + patchSize]
            overlapError = getOverlapError(srcPatch, syn, patchSize, ovSize, x, y)  
            textureMapPatch = textureMap[i:i + row, j:j + col]

            patchError = np.sum(np.power(targetMapPatch - textureMapPatch, 2))
            gradientError = np.sum(np.abs(edge_gradient - computeEdgeGradient(textureMapPatch)))

            lambda1, lambda2, lambda3 = alpha, (1 - alpha) / 2, (1 - alpha) / 2
            Errors[i, j] = lambda1 * overlapError + lambda2 * patchError + lambda3 * gradientError

    min_index = np.unravel_index(np.argmin(Errors), Errors.shape)
    a, b = min_index[:2]

    return texture[a:a+row, b:b+col]

def firstPatchSelect(texture,textureMap,targetMap,patchSize):
    H,W = textureMap.shape[0:2]
    Errors = np.zeros((H - patchSize, W - patchSize))  # 存储误差的矩阵
    targetMapPatch = targetMap[: patchSize, :patchSize]  # 从目标图区当前扫描到的块

    for i in range(H - patchSize):
        for j in range(W - patchSize): ## 遍历模糊后的texture上的块
            # srcPatch = texture[i + patchSize, j + patchSize]  # 原图块
            srcMapPatch = textureMap[i:i + patchSize, j:j + patchSize]

            e = np.sum(np.power(targetMapPatch - srcMapPatch, 2))
            Errors[i, j] = e

    print('First patch error:', Errors[:3])
    min_index = np.unravel_index(np.argmin(Errors), Errors.shape)
    print('First patch index:', min_index)
    a, b = min_index[0:2]
    return texture[a:a + patchSize, b:b + patchSize]

def transfer(texture,target,patchSize,alpha = 0,time = 0,syn_pre = None,enhanced = False):
    textureMap = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    targetMap = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)  # 转为灰度图

    textureMap = cv2.GaussianBlur(textureMap, (3,3), 0)
    targetMap = cv2.GaussianBlur(targetMap, (3,3), 0) #模糊
    ovSize = patchSize//6
    # print('overlap', ovSize)

    H, W = target.shape[:2]
    print('H,W,patchSize, overlap:',H,W,patchSize,ovSize)
    PatchNumRow = math.ceil((H - patchSize)/(patchSize - ovSize)) + 1
    PatchNumCol = math.ceil((W - patchSize)/(patchSize - ovSize)) + 1
    print("需要块数量：", PatchNumRow, PatchNumCol)
    print('time:', time)
    if time == 0: #判断是否为第一次迭代
        syn = np.zeros_like(target)
        # print('syn',syn.shape)
    else:
        syn = syn_pre

    for i in range(PatchNumRow-1):
        for j in range(PatchNumCol-1):
            # print('i,j:', i, j)
            x = j*(patchSize - ovSize)  # 扫描到的块第一个像素坐标
            y = i*(patchSize - ovSize)
            print('坐标(x,y):',x,y)
            if i == 0 and j == 0:
                patch = firstPatchSelect(texture, texture, target, patchSize)
            elif time == 0:   # 第一次运行时仅找块，不缝合   TODO:
                if enhanced:
                    patch = getBestPatch_enhanced(texture, texture, target, syn, x, y, patchSize, ovSize, alpha)
                else:
                    patch = getBestPatch(texture, texture, target, syn, x, y, patchSize, ovSize, alpha)
                print('mode: 2')

            else:  # 第一次之后的迭代，有缝合
                if enhanced:
                    patch = getBestPatch_enhanced(texture, texture, target, syn, x, y, patchSize, ovSize, alpha)
                else:
                    patch = getBestPatch(texture, texture, target, syn, x, y, patchSize, ovSize, alpha)
                patch = minCostPatch(patch, patchSize, ovSize, syn, y, x)
                print('mode: 3')
            print('Patch Size:', patch.shape)
            row, col = patch.shape[:2]
            syn[y:y + row, x:x + col] = patch
    return syn


def iteration(texture,target,patchSize,N,outputNum,enhanced = False):
    # texture = np.array(texture)
    # target = np.array(target)

    texture = texture.astype(np.float32)
    target = target.astype(np.float32)  # 转为float数据类型方便计算

    syn = transfer(texture, target, patchSize,enhanced=enhanced) #只拼接，不缝合
    cv2.imwrite('../output/transfer10_n0.jpg', syn)

    for i in range(1, N):
        patchSize = math.ceil(patchSize * (1-1/3)**(i-1)) #每次调整块大小
        alpha = 0.8 * (i-1)/(N-1) + 0.1
        print('迭代参数:',i,patchSize,alpha)
        syn = transfer(texture, target, patchSize, alpha, i, syn,enhanced=enhanced) #对前一次处理
        if enhanced:
            cv2.imwrite('../output/transfer'+str(outputNum)+'_enhanced_n' + str(i) + '.jpg', syn)
        else:
            cv2.imwrite('../output/transfer'+str(outputNum)+'_n' + str(i) + '.jpg', syn)

    return syn


for i in range(4):
    texture1 = cv2.imread('../src/texture/text'+str(i+1)+'.jpg') # 读取纹理图
    target1 = cv2.imread('../src/target/target5.jpg') # 读取目标图
    print('shape:', texture1.shape,target1.shape)
    syn1 = iteration(texture1,target1,25,3,i+1,enhanced=False)
    cv2.imwrite('../output/trans'+str(i+1)+'.jpg', syn1)
    syn2 = iteration(texture1,target1,25,3,i+1,enhanced=True)
    cv2.imwrite('../output/trans'+str(i+1)+'enhanced'+'.jpg', syn2)






















