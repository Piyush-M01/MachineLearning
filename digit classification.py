import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()
print(len(X_train))

print(len(X_test))

print(X_train[0].shape)

print(y_test)
#plt.matshow(X_train[0])
#plt.show()


#Scaling the values
X_train=X_train/255
X_test=X_test/255

#converting the 28 x 28 grid to 28**2 x 1
x_train_flattened=X_train.reshape(len(X_train),28*28)
x_test_flattened=X_test.reshape(len(X_test),28*28)

#creating simple neural network
model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flattened,y_train,epochs=5)


#evaluating accuracy on test dataset
print(model.evaluate(x_test_flattened,y_test))

#plt.matshow(X_test[0])
#plt.show()

predicted=model.predict(x_test_flattened)

print(np.argmax(predicted[0]))