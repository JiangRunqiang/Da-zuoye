from grad import *
from numpy import *

def loadData():
    train_x = []
    train_y = []
    fileIn = open('test.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

#读取测试集
train_x, train_y = loadData()
test_x = train_x;
test_y = train_y

#计算回归系数
opts = {'alpha': 0.01, 'maxIter': 1000, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

#在测试集上进行测试
accuracy = testLogRegres(optimalWeights, test_x, test_y)

#显示结果
print('分类的准确度为: %.3f%%' % (accuracy * 100))
showLogRegres(optimalWeights, train_x, train_y)

