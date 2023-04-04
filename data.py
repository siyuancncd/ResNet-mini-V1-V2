import utils_paths
import random
import numpy as np
from sklearn.model_selection import train_test_split
import skimage.io as io
import os
from sklearn import preprocessing
from keras.utils import to_categorical

def get_TrainData(path):
    print("---------start reading data---------")
    
    data = []
    labels  = []

    imagePaths = sorted(list(utils_paths.list_images(path)))
    #生成同一个随机数
    random.seed(42)
    #随机打乱数据集
    random.shuffle(imagePaths)
    
    #获取图像和图像的标签
    for imagePath in imagePaths:
        #读取图像数据
        img = io.imread(imagePath)
        img = np.reshape(img, (100,100,3))
        img= img / 255
        data.append(img)
        #读取标签
        label = imagePath.split(os.path.sep)[-2] #以路径分隔符分开，形成列表，然后取倒数第二个值
        labels.append(label)


    #把含有图片的列表转换为numpy格式，并且做scale操作
    data = np.array(data, dtype = 'float32')
    #把标签转换为numpy格式
    labels = np.array(labels)

    (trainX, val_X, trainY, val_Y) = train_test_split(data, labels, train_size = 0.75, test_size = 0.25, random_state = 42)#random_state是划分标记，如果值相等则划分结果也相同

    #  将标签矩阵二值化
    lb = preprocessing.LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    trainY = to_categorical(trainY)
    val_Y = lb.fit_transform(val_Y)
    val_Y = to_categorical(val_Y)
    return trainX, val_X, trainY, val_Y, len(lb.classes_)

def get_TestData(path):
    print("---------start reading data---------")
    data = []
    labels  = []

    imagePaths = sorted(list(utils_paths.list_images(path)))
    #生成同一个随机数
    random.seed(42)
    #随机打乱数据集
    random.shuffle(imagePaths)
    
    #获取图像和图像的标签
    for imagePath in imagePaths:
        #读取图像数据
        img = io.imread(imagePath)
        img = np.reshape(img, (100,100,3))#将图片转化为100*100*3
        img= img / 255
        data.append(img)
        #读取标签
        label = imagePath.split(os.path.sep)[-2] #以路径分隔符分开，形成列表，然后取倒数第二个值
        labels.append(label)


    #把含有图片的列表转换为numpy格式，并且做scale操作
    data = np.array(data, dtype = 'float32')
    #把标签转换为numpy格式
    labels = np.array(labels)

    test_X = data
    test_y = labels

    #  将标签矩阵二值化
    lb = preprocessing.LabelBinarizer()
    test_y = lb.fit_transform(test_y)
    test_y = to_categorical(test_y)
    return test_X, test_y