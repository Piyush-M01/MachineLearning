import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split


data=pd.read_csv('suv_data.csv')
data=data.drop('Gender',axis=1)

print(data.head())

#train,test,split
X_train,X_test,y_train,y_test=train_test_split(data[['Age','EstimatedSalary']],data.Purchased,test_size=0.2,random_state=2)

trainx=X_train.copy()
trainx['Age']=trainx['Age']/100
trainx['EstimatedSalary']=trainx['EstimatedSalary']/100000

testx=X_test.copy()
testx['Age']=testx['Age']/100
testx['EstimatedSalary']=testx['EstimatedSalary']/100000

#neural network

model=keras.Sequential([
keras.layers.Dense(2,input_shape=(2,),activation='relu',kernel_initializer='ones',bias_initializer='zeros'),
    keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print(model.fit(trainx,y_train,epochs=1500))


#evaluation on test data set
print(model.evaluate(testx,y_test))
print(model.predict(testx))
