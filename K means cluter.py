import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data=pd.read_csv('/home/piyush/Downloads/income.csv')
#print(data.head())

#plt.scatter(data['Age'],data['Income($)'])
#plt.show()

km=KMeans(n_clusters=3)
print(km)
'''
predicted=km.fit_predict(data[['Age','Income($)']])
print(predicted)

data['cluster']=predicted
print(data.head())

df1=data[data.cluster==0]
df2=data[data.cluster==1]
df2=data[data.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df1.Age,df1['Income($)'],color='red')
plt.scatter(df1.Age,df1['Income($)'],color='black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
'''

#since there is some problem with clustering, we need to scale the data
scaler=MinMaxScaler()
scaler.fit(data[['Income($)']])
data['Income($)']=scaler.transform(data[['Income($)']])

scaler.fit(data[['Age']])
data.Age=scaler.transform(data[['Age']])
predicted=km.fit_predict(data[['Age','Income($)']])
print(predicted)

data['cluster']=predicted
print(data.head())

df1=data[data.cluster==0]
df2=data[data.cluster==1]
df2=data[data.cluster==2]
'''
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df1.Age,df1['Income($)'],color='red')
plt.scatter(df1.Age,df1['Income($)'],color='black')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
'''

print(km.cluster_centers_)

k_range=range(1,10)
sse=[]
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(data[['Age','Income($)']])
    sse.append(km.inertia_)

print(sse)

#elbow plt
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_range,sse)
plt.show()
