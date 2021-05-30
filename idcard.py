import cv2 as cv
import numpy as np


def grayImg(img):
    # 转化为灰度图
    gray = cv.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    # otsu二值化操作
    retval, gray = cv.threshold(gray, 120, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
    return gray


def preprocess(gray):
    # 二值化操作，但与前面grayimg二值化操作中不一样的是要膨胀选定区域所以是反向二值化
    ret, binary = cv.threshold(gray, 180, 255, cv.THRESH_BINARY_INV)
    ele = cv.getStructuringElement(cv.MORPH_RECT, (15, 10))
    # 膨胀操作
    dilation = cv.dilate(binary, ele, iterations=1)
    return dilation


def findTextRegion(img):
    region = []
    # 1. 查找轮廓
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 300):
            continue
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        # 找到最小的矩形，该矩形可能有方向
        rect = cv.minAreaRect(cnt)
        # 函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
        # box是四个点的坐标
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue
        # 太扁的也不要
        if (height * 18 < width):
            continue
        if (width > img.shape[1] / 2 and height > img.shape[0] / 20):
            region.append(box)
    return region


def detect(img):
    gray = cv.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
    coefficients = [0, 1, 1]
    m = np.array(coefficients).reshape((1, 3))
    gray = cv.transform(gray, m)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        h = abs(box[0][1] - box[2][1])
        w = abs(box[0][0] - box[2][0])
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        y1 = min(Ys)
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)
        if w > 0 and h > 0 and x1 < gray.shape[1] / 2:
            idImg = cv.bitwise_not(grayImg(img[y1:y1 + h, x1:x1 + w]))
            return idImg

def CutIdCard(imgPath):
    img = cv.imread(imgPath, cv.IMREAD_COLOR)
    img = cv.resize(img, (428, 270), interpolation=cv.INTER_CUBIC)
    idImg = detect(img)
    idImg = cv.resize(idImg, (28*18, 28), interpolation=cv.INTER_CUBIC)
    contours, heriachy = cv.findContours(idImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    xs, ys, ws, hs = [], [], [], []
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)  # 外接矩形
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

    inds = np.array(xs).argsort()
    xs = np.array(xs)[inds]
    ys = np.array(ys)[inds]
    ws = np.array(ws)[inds]
    hs = np.array(hs)[inds]
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        w = ws[i]
        h = hs[i]
        idImg0 = cv.resize(idImg[y:y + h, x:x + w], (12, 20), interpolation=cv.INTER_AREA)
        tmpImg = np.zeros((28, 28))
        tmpImg[4:24, 8:20] = idImg0
        cv.imwrite("tmp/pic/"+str(i)+".png", tmpImg)

