# Classification Trees
# 1. Import the estimator object (model)
from sklearn.tree import DecisionTreeClassifier

# 2. Create an instance of the estimator
class_tree = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=10)

# 3. Use the trainning data to train the estimator
class_tree.fit(x_train, y_train)

# 4. Evalute the model
y_pred_test = class_tree.predict(x_test)
metrics.loc['accuracy','ClassTree'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','ClassTree'] = precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','ClassTree'] = recall_score(y_pred=y_pred_test,y_true=y_test)
# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)
# Naive Bayes Classifier
# 1. Import the estimator object (model)
from sklearn.naive_bayes import GaussianNB
#l0
# 2. Create an instance of the estimator
NBC = GaussianNB()

# 3. Use the trainning data to train the estimator
NBC.fit(x_train, y_train)

# 4. Evalute the model
y_pred_test = NBC.predict(x_test)
metrics.loc['accuracy','NaiveBayes'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NaiveBayes'] = precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NaiveBayes'] = recall_score(y_pred=y_pred_test,y_true=y_test)
# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
CMatrix(CM)
100*metrics
fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='bar',ax=ax)
ax.grid();
precision_nb, recall_nb, thresholds_nb = precision_recall_curve(y_true=y_test,
                                                                probas_pred=NBC.predict_proba(x_test)[:,1])
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_true=y_test,
                                                               probas_pred=logistic_regression.predict_proba(x_test)[:,1])
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(precision_nb, recall_nb, label='NaiveBayes')
ax.plot(precision_lr, recall_lr, label='LogisticReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
#ax.hlines(y=0.5, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();
# Confusion matrix for modified Logistic Regression Classifier
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(thresholds_lr[:],precision_lr[1:],label='Precision')
ax.plot(thresholds_lr, recall_lr[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel(' Precision Recall')
ax.set_title('Logistic Regression Classifier:Precision-Recall')
ax.hlines(y=0.6, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();
# Classifier with threshold of 0.2
y_pred_proba = logistic_regression.predict_proba(x_test)[:,1]
y_pred_test = (y_pred_proba >= 0.2).astype('int')
# confusion matrix
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print("Recall: ",100*recall_score(y_pred=y_pred_test,y_true=y_test))
print("Precision: ",100*precision_score(y_pred=y_pred_test,y_true=y_test))
CMatrix(CM)
