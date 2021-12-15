import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
titanic = sns.load_dataset('titanic')

test = sns.load_dataset('titanic')
a = pd.get_dummies(['a','b','c','d','e','f','g'])       #onehot 인코딩(컬럼과 인덱스가 겹치는 부분에 1.) 컬럼의 양을 늘릴 수 있다.
# 1. normalize - 백분율로 (데이터 - x.min())/ (x.max() - x.min())
# 2. standardization - z-score활용, 정규분포를 
# def normalization(x):
#     return (x- x.min())/ (x.max()-x.min())
def z_score(x):
     return (x - x.mean()) / x.std()

# b = titanic.apply(normalization)

# Machine Learing
x_df = titanic.drop('survived', axis =1)
y_df = titanic.survived
# 학습 데이터와 검증 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=12)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()

dt_clf.fit(x_train,y_train)
rf_clf.fit(x_train,y_train)
lr_clf.fit(x_train,y_train)

dt_pred = dt_clf.predict(x_test)
rf_pred = rf_clf.predict(x_test)
lr_pred = lr_clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,dt_pred))
print(accuracy_score(y_test,rf_pred))
print(accuracy_score(y_test,lr_pred))


# DecisionTreeClassifier()

# parameters = {'max_depth':[0,1,2,3,4,5,6,7,8,9,10],
# 'min_samples_split':[2,3,4,5,6,7,8,9,10],
# 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}

grid = sklearn.model_selection.GredSerchCV(dt_clf, param_grid+parameters,scoring='accuracy', cv=5, n_jobs=-1)

grid.fix(x_train, y_train)


best_dtclf = grid.best_estimator_
best_predictions = best_dtclf.predict(x_test)
accuracy_score(y_test, best_predictions)
# exec_kfold(best_dtclf)
# accuray_score(titanic.survived.values, titanic.sex.values)
# sklearn.metrics.confusion_matrix(y_test, best_predictions)

from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, best_predictions), columns=['0','1'])
# 정확도 : (TN + TP) / (TN + TP + FN + FP)  -> 정확도를 구한다
# Precision(정밀도) : TP / (TP + FP) -> 전체 가능도 중에 맞춘 것, 분모가 실제 positive 인 전체 수, FN을 낮추는 것
# Recall (재현율) : TP / (FN + TP) -> 분모가 positivie 라고 예측한 전체 수, FP를 낮추는 것




from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

def clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)

    print('--------------Confusion Matrix-----------------')
    print('confusion')
    print('정확도:{0:.04f}, 정밀도:{1:.4f}, 재현율:{2:.4f}, F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy,precision,recall,f1,auc))


clf_eval(y_test, best_predictions)

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.5)
pre_proba_ = lr_clf.predict_proba(x_test)[:,1].reshape(-1,1)
binarizer.fit_transform([[1,2,2,1,2,2.9,2.8,3,3.1,5]])

thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def clf_eval_thre(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold=i).fit(pred_proba)
        custom_predict = binarizer.transform(pred_proba)
        print('threshold:', i)
        clf_eval(y_test, custom_predict)

clf_eval_thre(y_test, lr_clf.predict_proba(x_test)[:, 1].reshape(-1,1))

from sklearn.metrics import precision_recall_curve
pred_proba_class = lr_clf.predict_proba(x_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class)
plt.figure(figsize=(20,10))
plt.plot(thresholds, precisions, label='precision')
plt.plot(thresholds, recalls, label='recall')
plt.show()

# F1 score, F1 = (2*(precision * recall)) / (precision + recall)

from sklearn.metrics import f1_score
f1_score(y_test, lr_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_teset, lr_pred)
accuracy_score(y_test, lr_pred)

clf_eval_thre(y_test, lr_clf.predict_proba(x_test)[:, 1].reshape(-1,1), thresholds)
