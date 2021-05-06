import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


data=pd.read_csv("suv_data.csv")

#print(data.head(5))

#data analyzing
sns.countplot(x="Purchased",hue="EstimatedSalary",data=data)

#data  wrangling
#print(data.isnull().sum())
data.drop("User ID",axis=1,inplace=True)

#removing all string values
data.drop("Gender",axis=1,inplace=True)
final=data
#print(final)

#training the data set
X=final.drop("Purchased",axis=1)
y=final["Purchased"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
LR=LogisticRegression()
LR.fit(X_train,y_train)

prediction=LR.predict(X_test)

#testing the code
print(accuracy_score(y_test,prediction))
#plt.scatter(y_test,prediction)
#plt.show()
