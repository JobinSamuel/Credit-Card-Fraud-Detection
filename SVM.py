print(len(x_train), len(x_test), len(y1_train), len(y_test))
from sklearn import svm
regressor = svm.SVC()
#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 5) 
regressor.fit(x1_train, y1_train)
y_pred = regressor.predict(x1_test)
#print("Confusion Matrix: ",confusion_matrix(y_pred.round(),y_test))
accuracy = accuracy_score( y_pred.round(),y1_test)
print("Support Vector Machine Accuracy ",accuracy)
