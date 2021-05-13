import pandas as pd

data=pd.read_csv('/home/piyush/Downloads/salaries.csv')
#print(data.head())

inputs=data.drop('salary_more_then_100k',axis=1)
target=data['salary_more_then_100k']

#print(inputs)
#print(target)

#converting labels to numbers

from sklearn.preprocessing import LabelEncoder
company=LabelEncoder()
job=LabelEncoder()
degree=LabelEncoder()

inputs['company_n']=company.fit_transform(inputs['company'])
inputs['job_n']=job.fit_transform(inputs['job'])
inputs['degree_n']=degree.fit_transform(inputs['degree'])

#print(inputs)

#drop label columns
input_n=inputs.drop(['company','job','degree'],axis=1)
#print(input_n)


#splitting dataset into training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(input_n[['company_n','job_n','degree_n']],target,test_size=0.2,random_state=100)


#training
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

score=model.score(input_n,target)
print(score)

prediction=model.predict(X_test)
print(prediction)