# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:40:07 2023

@author: jk
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ZeroPadding1D ,BatchNormalization
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, GRU, Bidirectional, TimeDistributed, Multiply, Lambda, Activation



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# 计算MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):

    true += epsion
    return np.sum(np.abs((true-pred)/true))/len(true)*100

def calcR2(true, pred):
    return r2_score(true, pred)


# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#超参数输入
look_back = 20 #输入历史时间步 安卓10，OpenStack 20
T = 1         #预测未来时间步
epochs = 100  #迭代次数
num_features= 10  #输入特证数
embed_dim = 32  #嵌入维度，要与filter的维度一致，OpenStack 32，安卓 16 
dense_dim= 64  #隐藏层神经元个数 相同 64
num_heads = 2   #头数
dropout_rate = 0.2  #失活率
num_blocks = 2  #编码器解码器数AAA
learn_rate = 0.001  #学习率
batch_size = 32 #批大小



def divideTrainTest(dataset, rate = 2613/(3044+2613)): 

    print (len(dataset))
  #  train_size = int(len(dataset) * (1-rate))
    test_size = int(len(dataset) * rate)
  
    train, test = dataset[: -test_size], dataset[-test_size:]#有时会需要-1
    print (len(train))
    print (len(test))       
    return train, test


def create_dataset(dataset, look_back):
    '''
    对数据进行处理,当使用先划分数据集时，分别修改一下就可以用了
    '''
      
    
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        # 如果你想加pollution，下面就不要动
        # 如果不加，下面改成X.append(data[i:i + size, 1:])，然后下面fea_num=7       
        a = dataset[i:(i+look_back),1:]   #[i:(i+look_back),0:10] 前11列 
        dataX.append(a)
    #    X.append(data[i:i + size, :])
        dataY.append(dataset[i + look_back, 0])  # (dataset[i + look_back, 12]) 第12列RUL
    
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data, dtype='float64')
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data, normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    
    #print ("ata22",data.shape)
    for i in range(0,(data.shape[1])):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data


#读取数据
dataset = pd.read_csv("./RUL.csv") 
                                                             # OpenStack o4t1 A, o4t2 B, o4t3 C


print(dataset.columns)
print(dataset.shape) 






train, test = divideTrainTest(dataset)
 
 
train, normalize = NormalizeMult(train)
test, normalize1 = NormalizeMult(test)


train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)
   
 
print("trainX shape is", train_X.shape)
print("trainY shape is", train_Y.shape)
print("testX shape is", test_X.shape)
print("testY shape is", test_Y.shape)

print(len(train_X))
print(train_X.shape,train_Y.shape)

print(len(test_X))
print(test_X.shape,test_Y.shape)



#定义TCN模块
def tcn_block(input_layer, n_filters, kernel_size, dilation_rate):
  
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same', activation='relu')(input_layer)
   
    dropout_layer = Dropout(dropout_rate)(conv_layer)
    # 残差连接
    Res = Conv1D(filters = n_filters, kernel_size=1, padding='same')(input_layer)
    residual_layer = Add()([Res, dropout_layer])
    
    
    return  residual_layer


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.dense1 = layers.Dense(dense_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)


####dense层
        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)
        return out2

class TransformerTimeSeriesModel(keras.Model):
    def __init__(self, num_blocks,embed_dim, dense_dim, num_heads, dropout_rate, output_sequence_length):
        super(TransformerTimeSeriesModel, self).__init__()

        self.embedding = layers.Dense(embed_dim, activation="relu")
        self.transformer_encoder = [TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.final_layer = layers.Dense(output_sequence_length)

    def call(self, inputs, training):
        x = self.embedding(inputs)
        for i in range(len(self.transformer_encoder)):
            x = self.transformer_encoder[i](x, training=training)

#        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
#        x = self.final_layer(x)
#        x = tf.reshape(x, [-1, T])
        return x

# 创建Transformer模型
transformer_model = TransformerTimeSeriesModel(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T)

# 创建多尺度TCN和transformer模型
input_shape = (look_back, num_features)
inputs = layers.Input(shape=input_shape)


'''
多尺度
'''
#### channel 1  安卓 filter 16，OpenStack 32
tcn1_1 = tcn_block(inputs, n_filters=32, kernel_size=2, dilation_rate=1)
tcn1_2 = tcn_block(tcn1_1, n_filters=32, kernel_size=2, dilation_rate=2)
#tcn1_3 = tcn_block(tcn1_2, n_filters=64, kernel_size=2, dilation_rate=4)

#### channel 2
tcn2_1 = tcn_block(inputs, n_filters=32, kernel_size=3, dilation_rate=1)
tcn2_2 = tcn_block(tcn2_1, n_filters=32, kernel_size=3, dilation_rate=2)
#tcn2_3 = tcn_block(tcn2_2, n_filters=64, kernel_size=3, dilation_rate=4)

#### channel 3
tcn3_1 = tcn_block(inputs, n_filters=32, kernel_size=5, dilation_rate=1)
tcn3_2 = tcn_block(tcn3_1, n_filters=32, kernel_size=5, dilation_rate=2)
#tcn3_3 = tcn_block(tcn3_2, n_filters=64, kernel_size=5, dilation_rate=4)


'''
transformer
'''

transformer_output = transformer_model(inputs, training=True)
#transformer_output = Conv1D(32, kernel_size=1, padding='same')(transformer_output)

'''
融合
'''
TCN = Add()([tcn1_2, tcn2_2, tcn3_2, transformer_output])  ### OpenStack的参数
#TCN = Concatenate()([tcn1_2, tcn2_2, tcn3_2, transformer_output])  # Concatenate 方式
'''
TTAF 预测
'''
fc = Dense(128, activation = 'relu')(TCN) 
fc = Flatten()(fc) #
output = Dense(1)(fc) #


model = keras.Model(inputs=inputs, outputs=output)

# 编译模型
opt = Adam(learning_rate=learn_rate)
model.compile(optimizer= opt, loss='mean_squared_error', metrics=["mean_absolute_percentage_error"])

model.summary()
# 训练模型

# 训练模型
history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs) #, validation_split=0.2
#loss画图，看训练是否过拟合了
#fig1 = plt.figure(figsize=(12, 8))
#plt.plot(history.history['loss'],label='train loss')
## plt.plot(history.history['val_loss'], label='val')
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch') 
#plt.show()

#模型在测试集进行预测 test_X, test_Y 
testPred = model.predict(test_X)
testPred = testPred.reshape(-1,1)
#label = test_Y.reshape(-1,1)

#测试集数据反归一化
testPred = FNormalizeMult(testPred, normalize1)
test_Y = test_Y.reshape(-1, 1)
test_Y = FNormalizeMult(test_Y, normalize1)

    # 绘制损失函数
  

# 计算模型的评价指标
MAE = calcMAE(test_Y, testPred)
print(MAE)
RMSE = calcRMSE(test_Y, testPred)
print(RMSE)
r2 = calcR2(test_Y, testPred)
print(r2)
          
    
    
          