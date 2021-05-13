import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


#print(dir(iris))

#converting iris into pandas dataset
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head())

df['target']=iris.target
#print(df.head())
#print(iris.target_names)

#print(df[df.target==1].head())

df['flower_name']=df.target.apply(lambda x: iris.target_names[x])
#print(df.head())

#visualisation
import matplotlib.pyplot as plt
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='.')
plt.show()


#training
from sklearn.model_selection import train_test_split
X=df.drop(['target','flower_name'],axis=1)
Y=df.target

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=100)

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

print(model.predict(X_test))
