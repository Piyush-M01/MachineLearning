from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits=load_digits()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.3)
'''
lr=LogisticRegression()
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))

svm=SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))

rf=RandomForestClassifier(n_estimators=40)
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
'''
#K FOLD
#method 1 of checking the scores
from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
#print(kf)


def get_scores(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

from sklearn.model_selection import StratifiedKFold
fold=StratifiedKFold(n_splits=3)
scores_l=[]
scores_svm=[]
scores_rf=[]


'''
for train_index, test_index in kf.split(digits.data):
    X_train,X_test,y_train,y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    scores_l.append(get_scores(LogisticRegression(),X_train,X_test,y_train,y_test))
    scores_svm.append(get_scores(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_scores(RandomForestClassifier(), X_train, X_test, y_train, y_test))
    print(scores_l , scores_svm , scores_rf)
'''


#method 2 of checking the scores
from sklearn.model_selection import cross_val_score
cvs=cross_val_score(LogisticRegression(),digits.data,digits.target)
cvs1=cross_val_score(SVC(),digits.data,digits.target)
cvs2=cross_val_score(RandomForestClassifier(n_estimators=40),digits.data,digits.target)
print(cvs,cvs1,cvs2)