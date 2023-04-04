from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import json
from data import get_TrainData, get_TestData
from model import resnet_mini_v1, resnet_mini_v2

def train(trainX, trainY,model, modelname,Epochs,LearningRate,Batch_Size):
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr = LearningRate, decay = LearningRate/Epochs), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(modelname+".hdf5", monitor='val_acc', verbose=1,save_best_only=True, save_weights_only=True)
    callbacks_list = [checkpoint]
    model.summary() #打印网络各层参数

    history = model.fit(trainX, trainY, batch_size=Batch_Size, epochs=Epochs, verbose=1, validation_data=(val_X, val_Y), callbacks = callbacks_list)

    ##保存训练日志
    json_str = json.dumps(history.history)
    with open((modelname + '.json'), 'w') as file_txt:
        file_txt.write(json_str)

def evaluate(model_file):
    classes = ["stage1", "stage2", "1ms", "4ms", "light", "middle", "heavy", "all"]
    path = 'D:\\Developer\\test\\'

    # model = load_model("distilled_resnet10_small_small.hdf5")
    ##如果有自定义层，则不能save_model而是先保存权重再加载，否则会导入模型失败
    model.load_weights(model_file + ".hdf5", by_name=False)# by_name是False就按网络拓扑加载，是True就按层名加载
    # model = load_model("mini_ResNet10_batch5_20epochs.hdf5")
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr = LearningRate, decay = LearningRate/Epochs), metrics=['accuracy'])
    
    for i in range(len(classes)):
        new_path = path + classes[i]
        testX, testY = get_TestData(new_path)
        score = model.evaluate(testX, testY, verbose=0)
        print("Test accuracy of " + classes[i] + " is ", score[1])
    
if __name__ == "__main__":

    path = 'D:\\Developer\\train' # 换成你的
    trainX, val_X, trainY, val_Y, classes = get_TrainData(path)

    Epochs = 50
    LearningRate = 1e-5
    Batch_Size = 5

    model = resnet_mini_v1()
    model_file = "./resnet_mini_v1_batch5_epochs50"    #保存weight就要加./，保存model不用

    train(trainX, trainY, model, model_file, Epochs,LearningRate,Batch_Size)
    # evaluate(model_file)
