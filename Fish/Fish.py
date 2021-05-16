import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('/home/piyush/Downloads/Fish/Fish.csv')
print(data.shape)

#finding the outliers
  #single variable outliers

#sns.boxplot(x=data['Weight'])#weight has 3 outliers after 1500
'''
sns.boxplot(x=data['Length1']) # 2 after
sns.boxplot(x=data['Length2']) # 2 after 55
sns.boxplot(x=data['Length3']) # 1 after 65
plt.show()

fig, ax =plt.subplots(figsize=(6,8))
ax.scatter(data['Weight'],data['Length1'],data['Length2'],data['Length3'])
ax.legend(['Weight','Lenght1','Length2','Length3'])
plt.show()
'''

species=LabelEncoder()
data['Species_n']=species.fit_transform(data['Species'])
label=species.classes_
print(label)

#removing outliers
data = data[~(data['Weight'] >= 1500)]
data=data[~(data['Length1']>=50)]
data=data[~(data['Length2']>=55)]
data=data[~(data['Length3']>=65)]
print(data.shape)

fig, ax =plt.subplots(figsize=(6,8))
ax.scatter(data['Weight'],data['Length1'],data['Length2'],data['Length3'])
ax.legend(['Weight','Lenght1','Length2','Length3'])
plt.show()

x=data.drop(['Species','Species_n'],axis=1)
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=data['Species_n']


from sklearn.model_selection import train_test_split
lr = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#print(X_test)
#print(y_test)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
R=lr.predict(X_test)
print(label[R])