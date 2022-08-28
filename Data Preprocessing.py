import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
# Data preparation)
#default = pd.read_excel('UCI_Credit_Card.csv',skiprows=[0],index_col="ID")
default = pd.read_csv('UCI_Credit_Card.csv',index_col="ID")
default.rename(columns=lambda x: x.lower(), inplace=True)
default.head()
# Base values: female, other_education,not_married
default['grad_school'] = (default['education'] == 1).astype('int')
default['university'] = (default['education'] == 2).astype('int')
default['high_school'] = (default['education'] == 3).astype('int')

default.drop('education' , axis=1, inplace=True)

default['male'] = (default['sex']==1).astype('int')
default.drop('sex' , axis=1, inplace=True)

default['married'] = (default['marriage']==1).astype('int')
default.drop('marriage' , axis=1, inplace=True)

#for pay features if the <=0 then it means it was not delayed
pay_features = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']
for p in pay_features:
    default.loc[default[p]<=0, p] = 0

default.rename(columns={'default payment next month':'default'},inplace=True)
default.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score,confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler
default.head()
target_name = 'default.payment.next.month'
x = default.drop(target_name, axis=1)
robust_scaler = RobustScaler()
x = robust_scaler.fit_transform(x)
y = default[target_name]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123, stratify=y)
def CMatrix(CM, labels=['pay' , 'default.payment.next.month']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name= 'TRUE'
    df.columns.name= 'PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df
# Data frame for evaluation metrics
metrics = pd.DataFrame(index=['accuracy','precision','recall'],
                      columns=['NULL','LogisticReg','ClassTree','NaiveBayes'])
# Null model
y_pred_test = np.repeat(y_train.value_counts().idxmax(), y_test.size)
metrics.loc['accuracy','NULL'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NULL'] = precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NULL'] = recall_score(y_pred=y_pred_test,y_true=y_test)

CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)
