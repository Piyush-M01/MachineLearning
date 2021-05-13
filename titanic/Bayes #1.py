import pandas as pd


data=pd.read_csv('/home/piyush/Downloads/titanic/train.csv')
#print(data.head())

data.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket','Embarked'],axis=1,inplace=True)
#print(data.head())

target=data.Survived
inputs=data.drop('Survived',axis=1)

dummy=pd.get_dummies(inputs.Sex)
#print(dummy.head())

inputs=pd.concat([inputs,dummy],axis=1)
inputs.drop('Sex',axis=1,inplace=True)
inputs=inputs.fillna(inputs.mean())
#inputs.Age=inputs.Age.fillna(inputs.Age.mean())

#print(inputs)

#training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)


#naives bayes model
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
#print(model.score(X_test,y_test))

test=pd.read_csv('/home/piyush/Downloads/titanic/test.csv')
test.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket','Embarked'],axis=1,inplace=True)
dum=pd.get_dummies(test.Sex)
test=test.drop(['Sex'],axis=1)
test=pd.concat([test,dum],axis=1)
test=test.fillna(test.mean())
#print(test)

print(model.predict(test))