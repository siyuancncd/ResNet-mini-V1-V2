from tensorflow.keras.layers import Add,Conv2D,BatchNormalization,Activation,Concatenate,Input,MaxPooling2D,AveragePooling2D,Flatten,Dense,Softmax
from tensorflow.keras.models import *

def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def resiidual_a_or_b(input_x, filters, flag):
    if flag == 'a':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 1, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        # 输出
        y = Add()([x, input_x])

        return y
    elif flag == 'b':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 2, input_x)
        x = Conv_BN_Relu(filters, (3, 3), 1, x)
        # 支路下采样
        input_x = Conv_BN_Relu(filters, (1, 1), 2, input_x)

        # 输出
        y = Add()([x, input_x])

        return y

def resiidual(input_x, filters):
    # 主路
    x1 = Conv_BN_Relu(filters, (3, 3), 2, input_x)
    x2 = Conv_BN_Relu(filters, (3, 3), 1, x1)
    # 支路下采样
    y = Conv_BN_Relu(filters, (1, 1), 2, input_x)

    # 输出
    out = Concatenate()([x1, x2, y])

    return out

def resnet_mini_v1(with_softmax=True):
    # 第一层
    input_shape = (100, 100, 3)
    input_layer = Input(shape=input_shape)
    conv1 = Conv_BN_Relu(8, (7, 7), 1, input_layer)
    conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)

    # conv2_x
    x = resiidual_a_or_b(conv1_Maxpooling, 8, 'b')

    # conv3_x
    x = resiidual_a_or_b(x, 16, 'b')

    # conv4_x
    x = resiidual_a_or_b(x, 32, 'b')

    # conv5_x
    x = resiidual_a_or_b(x, 64, 'b')

    # 最后一层
    x = AveragePooling2D(padding="same")(x)
    x = Flatten()(x)
    x = Dense(2)(x)
    #x = Dropout(0.5)(x)
    if with_softmax:
        x = Softmax(axis=-1)(x)
    output_layer = x

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model

def resnet_mini_v2(with_softmax=True):
    # 第一层
    input_shape = (100, 100, 3)
    input_layer = Input(shape=input_shape)
    conv1 = Conv_BN_Relu(8, (7, 7), 1, input_layer)
    conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)

    # conv2_x
    x = resiidual(conv1_Maxpooling, 8)

    # conv3_x
    x = resiidual(x, 16)

    # conv4_x
    x = resiidual(x, 32)

    # conv5_x
    x = resiidual(x, 64)

    # 最后一层
    x = AveragePooling2D(padding="same")(x)
    x = Flatten()(x)
    x = Dense(2)(x)
    if with_softmax:
        x = Softmax(axis=-1)(x)
    output_layer = x

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
