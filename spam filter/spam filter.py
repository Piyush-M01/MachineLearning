import pandas as pd
import numpy as np

data=pd.read_csv('/home/piyush/Downloads/spam filter/emails.csv')
#print(data.head())

#print(data.groupby('spam').describe())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.text,data.spam,test_size=0.2)

#removing string emails
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
X_train_count=v.fit_transform(X_train.values)
#print(v.get_feature_names())
#print(X_train_count.toarray())

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_count,y_train)

X_test_count=v.transform(X_test)
print(model.score(X_test_count,y_test))

#method 2

from sklearn.pipeline import Pipeline
clf=Pipeline([('vectorizer',CountVectorizer()),('nb',MultinomialNB())])
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
