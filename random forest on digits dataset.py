import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
digits=load_digits()

import matplotlib.pyplot as plt

'''
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
    plt.show()
'''

df=pd.DataFrame(digits.data)
#print(df)
#print(digits.target)


df['target']=digits.target
#print(df.head())

#train test split
from sklearn.model_selection import train_test_split
X=df.drop(['target'],axis=1)
Y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)


#random forest classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)

#score
print(model.score(X_test,y_test))
print(model.predict(X_test))

#analysing the model
#method1
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,model.predict(X_test))
print(cm)

#method2
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
