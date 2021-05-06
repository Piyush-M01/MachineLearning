#house price
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

read=pd.read_csv('USA_Housing.csv')
#print(read.info)
#print(read.describe())
#print(read.columns)
X=read[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=read[['Price']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
ln=LinearRegression()
ln.fit(X_train,y_train)


prediction=ln.predict(X_test)

plt.scatter(y_test,prediction)
plt.show()
#sns.distplot((y_test-prediction),bins=50)
#plt.show()
