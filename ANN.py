pay.shape
print(len(x_train), len(x_test), len(y_train), len(y_test))

x1_train = x_train
x1_test = x_test
y1_train = y_train
y1_test = y_test
# ANN Implementation with Perceptron
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

p = Perceptron(random_state=42,
              max_iter=10,
              tol=0.001)
p.fit(x_train, y_train)
y_pred = p.predict(x_test)
confusion_matrix(y_pred=y_pred,y_true=y_test)
ann_accuracy = accuracy_score(y_pred=y_pred,y_true=y_test)
print('ANN Accuracy = ',ann_accuracy)
