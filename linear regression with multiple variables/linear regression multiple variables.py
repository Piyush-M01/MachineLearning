import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#reading
data=pd.read_csv('Housing.csv')

#cleaning
datas=data.drop(['mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus'],axis=1)
# (1)checking for null
#print(datas.isnull().sum())

#training
scaler=StandardScaler()
scaled=scaler.fit_transform(datas)
#print(datas)

X=datas[['area','bedrooms','bathrooms','stories']]
y=datas.price


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

lm=LinearRegression()
lm.fit(X_train,y_train)



print(lm.coef_)
print(lm.intercept_)

prediction=lm.predict(X_test)
r2_score = lm.score(X_test,y_test)
print(r2_score*100,'%')
plt.scatter(y_test,prediction)
plt.show()
print(prediction)