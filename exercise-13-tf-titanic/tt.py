'''kaggle数据titanic生存预测
方法：tensorflow 实现 对数几率回归（逻辑回归）'''

import numpy as np
import pandas as pd

'''训练集读数据+预处理'''
df=pd.read_csv('./titanic/train.csv') #(891,12)

#把PassengerId、Name和Ticket这三列数据去掉
data=df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

#把年龄中NAN值用均值替换
data['Age'].fillna(data['Age'].mean(),inplace=True)

#pd.factorize()把字符型变量分解为数字，注意与pd.get_dummies()哑变量的one-hot编码的区别
#Cabin这一列中也有NAN值，因式分解被读取为-1
data['Cabin']=pd.factorize(data.Cabin)[0]

#偷个懒，其他的列中的NAN值都用0填充(不过数据只有年龄有nan值，所以这一步没啥必要就是)
data.fillna(0,inplace=True)

data['Sex']=[1 if x=='male' else 0 for x in data.Sex]

data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']

data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']

train_data=data[['Sex','Age','SibSp','Parch','Fare','Cabin','p1','p2','p3','e1','e2','e3']]
# print (train_data) #(891,12)
#      Sex        Age  SibSp  Parch      Fare  Cabin  p1  p2  p3  e1  e2  e3
# 0      1  22.000000      1      0    7.2500     -1   0   0   1   1   0   0
# 1      0  38.000000      1      0   71.2833      0   1   0   0   0   1   0
# 2      0  26.000000      0      0    7.9250     -1   0   0   1   1   0   0
# 3      0  35.000000      1      0   53.1000      1   1   0   0   1   0   0
train_target=data[['Survived']]
# print (train_target) #(891,1)
#      Survived
# 0           0
# 1           1
# 2           1

'''测试集读取+预处理'''
test=pd.read_csv('./titanic/test.csv')
test=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
test['Sex']=np.array(test['Sex']=='male').astype(int)
test['Age']=test['Age'].fillna(test['Age'].mean())
test['Cabin']=pd.factorize(test['Cabin'])[0]
test=pd.concat([test,pd.get_dummies(test['Pclass'],prefix='P')],axis=1)
test=pd.concat([test,pd.get_dummies(test['Embarked'],prefix='E')],axis=1)
del test['Pclass']
del test['Embarked']

test_data=test
#    Sex   Age  SibSp  Parch     Fare  Cabin  P_1  P_2  P_3  E_C  E_Q  E_S
# 0    1  34.5      0      0   7.8292     -1    0    0    1    0    1    0
# 1    0  47.0      1      0   7.0000     -1    0    0    1    0    0    1
# 2    1  62.0      0      0   9.6875     -1    0    1    0    0    1    0
# 3    1  27.0      0      0   8.6625     -1    0    0    1    0    0    1
# 4    0  22.0      1      1  12.2875     -1    0    0    1    0    0    1
test_target=pd.read_csv('./titanic/gender_submission.csv')[['Survived']]
# print (test_data.shape,test_target.shape) #(418, 12) (418, 1)

'''搭建神经网络'''

import tensorflow as tf
x=tf.placeholder(tf.float32,shape=[None,12])
y=tf.placeholder(tf.float32,shape=[None,1])

weight=tf.Variable(tf.random_normal([12,1]))
bias=tf.Variable(tf.random_normal([1]))
output=tf.matmul(x,weight)+bias

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=output))
pred=tf.cast(tf.sigmoid(output)>0.5,tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(pred,y),tf.float32))

train_step=tf.train.GradientDescentOptimizer(0.003).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

epoch=25000
train_loss=[]
train_accuracy=[]
test_accuracy=[]
for i in range(epoch):
    #打乱样本数据的顺序
    #df=df.sample(frac=1) 这种方法比较简便！但是train_data的打乱顺序要和train_target保持一致，所以选👇的方法
    shuffledindex=np.random.permutation(len(train_data))
    train_data=train_data.iloc[shuffledindex]
    train_target=train_target.iloc[shuffledindex]
    
    #batchsize=100,小批量梯度下降，9个批次所以更新9次参数
    for n in range(len(train_data)//100 +1):
        batch_train_data=train_data[100*n:100*(n+1)]
        batch_train_target=train_target[100*n:100*(n+1)]
        sess.run(train_step,feed_dict={x:batch_train_data,y:batch_train_target})
    
    #每完成1000次epoch后，记录损失函数和accuracy
    #loss_temp和train_accuracy_temp用的是batch数据跑起来都好慢，花了差不多10分钟....😢
    #其实batchsize应该是越大越好的，可是小穷鬼的硬件水平有限...所以batchsize为100、200
    if i%1000==0:
        loss_temp=sess.run(loss,feed_dict={x:batch_train_data,y:batch_train_target})
        train_loss.append(loss_temp)

        train_accuracy_temp=sess.run(accuracy,feed_dict={x:batch_train_data,y:batch_train_target})
        train_accuracy.append(train_accuracy_temp)

        test_accuracy_temp=sess.run(accuracy,feed_dict={x:test_data,y:test_target})
        test_accuracy.append(test_accuracy_temp)

        print (loss_temp,train_accuracy_temp,test_accuracy_temp)

# 40.97089 0.35164836 0.354067
# 0.6787327 0.6703297 0.6267943
# 0.6683779 0.7032967 0.64593303
# 0.6278029 0.6813187 0.6650718
# 0.6131146 0.6703297 0.67942584
# 0.48971012 0.7802198 0.68899524
# 0.4659122 0.83516484 0.70095694
# 0.52839583 0.7692308 0.71291864
# 0.52593213 0.7802198 0.73444974
# 0.5402897 0.7802198 0.74880385
# 0.4179724 0.8021978 0.79904306
# 0.47871646 0.7692308 0.81578946
# 0.5282071 0.7692308 0.83253586
# 0.40247458 0.85714287 0.8516746
# 0.5404938 0.7692308 0.8636364
# 0.45501143 0.7912088 0.8779904
# 0.4387093 0.7802198 0.8803828
# 0.4883889 0.7802198 0.8803828
# 0.35613993 0.8681319 0.8851675
# 0.53591096 0.72527474 0.8899522
# 0.44170815 0.7802198 0.8995215
# 0.379686 0.85714287 0.90909094
# 0.43209836 0.7912088 0.91626793
# 0.5008872 0.74725276 0.9282297
# 0.47590503 0.7692308 0.930622

'''作图'''
import matplotlib.pyplot as plt
plt.plot(train_loss,'k-')
plt.title('train_loss')
plt.show()

plt.plot(train_accuracy,'b-',label='train_accuracy')
plt.plot(test_accuracy,'r--',label='test_accuracy')
plt.title('accuracy: train(batch) VS. test')
plt.legend()
plt.show()