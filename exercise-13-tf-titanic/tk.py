'''kaggle数据titanic生存预测
方法：tensorflow的【keras】 实现 对数几率回归（逻辑回归）'''

import numpy as np
import pandas as pd

'''训练集读数据+预处理'''
df=pd.read_csv('/Users/zhangying/Desktop/tf-exercise/titanic/train.csv') #(891,12)
data=df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Cabin']=pd.factorize(data.Cabin)[0]
data['Sex']=[1 if x=='male' else 0 for x in data.Sex]
data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']
data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']

train_data=data[['Sex','Age','SibSp','Parch','Fare','Cabin','p1','p2','p3','e1','e2','e3']] #(891,12)
#      Sex        Age  SibSp  Parch      Fare  Cabin  p1  p2  p3  e1  e2  e3
# 0      1  22.000000      1      0    7.2500     -1   0   0   1   1   0   0
# 1      0  38.000000      1      0   71.2833      0   1   0   0   0   1   0
# 2      0  26.000000      0      0    7.9250     -1   0   0   1   1   0   0
# 3      0  35.000000      1      0   53.1000      1   1   0   0   1   0   0
train_target=data[['Survived']] #(891,1)



'''keras搭建神经网络'''
from tensorflow import keras
from tensorflow.keras import layers
model=keras.Sequential()
model.add(layers.Dense(1,input_dim=12,activation='sigmoid'))
# model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
hist=model.fit(train_data,train_target,epochs=300)
#print(hist.history) #字典 （不妨通过hist.history.keys()查看有哪些键 dict_keys(['loss','acc])      ）

import matplotlib.pyplot as plt
plt.plot(range(300),hist.history.get('loss'))
plt.show()


'''tensorflow搭建神经网络'''

# import tensorflow as tf
# x=tf.placeholder(tf.float32,shape=[None,12])
# y=tf.placeholder(tf.float32,shape=[None,1])

# weight=tf.Variable(tf.random_normal([12,1]))
# bias=tf.Variable(tf.random_normal([1]))
# output=tf.matmul(x,weight)+bias

# loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=output))
# pred=tf.cast(tf.sigmoid(output)>0.5,tf.float32)
# accuracy=tf.reduce_mean(tf.cast(tf.equal(pred,y),tf.float32))

# train_step=tf.train.GradientDescentOptimizer(0.003).minimize(loss)

# sess=tf.Session()
# sess.run(tf.global_variables_initializer())

# epoch=25000
# train_loss=[]
# train_accuracy=[]
# test_accuracy=[]
# for i in range(epoch):
#     #打乱样本数据的顺序
#     #df=df.sample(frac=1) 这种方法比较简便！但是train_data的打乱顺序要和train_target保持一致，所以选👇的方法
#     shuffledindex=np.random.permutation(len(train_data))
#     train_data=train_data.iloc[shuffledindex]
#     train_target=train_target.iloc[shuffledindex]
    
#     #batchsize=100,小批量梯度下降，9个批次所以更新9次参数
#     for n in range(len(train_data)//100 +1):
#         batch_train_data=train_data[100*n:100*(n+1)]
#         batch_train_target=train_target[100*n:100*(n+1)]
#         sess.run(train_step,feed_dict={x:batch_train_data,y:batch_train_target})
    
#     #每完成1000次epoch后，记录损失函数和accuracy
#     #loss_temp和train_accuracy_temp用的是batch数据跑起来都好慢，花了差不多10分钟....😢
#     #其实batchsize应该是越大越好的，可是小穷鬼的硬件水平有限...所以batchsize为100、200
#     if i%1000==0:
#         loss_temp=sess.run(loss,feed_dict={x:batch_train_data,y:batch_train_target})
#         train_loss.append(loss_temp)

#         train_accuracy_temp=sess.run(accuracy,feed_dict={x:batch_train_data,y:batch_train_target})
#         train_accuracy.append(train_accuracy_temp)

#         test_accuracy_temp=sess.run(accuracy,feed_dict={x:test_data,y:test_target})
#         test_accuracy.append(test_accuracy_temp)

#         print (loss_temp,train_accuracy_temp,test_accuracy_temp)
# sess.close()

