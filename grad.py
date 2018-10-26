from numpy import *
import matplotlib.pyplot as plt
import time
#计算sigmod函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#回归系数计算函数
def trainLogRegres(train_x, train_y, opts):
    # 得到计算起始时间
    startTime = time.time()

    numSamples, numFeatures = shape(train_x)
    alpha = opts['alpha'];#得到控制台输入的初始回归系数
    maxIter = opts['maxIter']#得到控制台输入的最大迭代次数
    weights = ones((numFeatures, 1))

    # 通过梯度下降法迭代回归系数
    for k in range(maxIter):
        output = sigmoid(train_x * weights)
        error = train_y - output
        weights = weights + alpha * train_x.transpose() * error
    return weights


# 测试，利用得到的回归系数进行分类，并与正确值进行比较，得到精确度
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# 通过matplotlib库进行绘图
def showLogRegres(weights, train_x, train_y):
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 3:
        return 1
    #通过循环绘出每个点，并对不同的类用不同的颜色画出
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    #通过回归系数画出分类线
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()