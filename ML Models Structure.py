import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os



class ClassifierModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test



    def ANN(self):
      from sklearn.neural_network import MLPClassifier
      ANN_Classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
      ANN_Classifier.fit(self.X,self.y)
      y_pred = ANN_Classifier.predict(self.X_test)

      print("\n")
      print("************************* Nueral Network Classifier ************************* \n")
      print('Classification Report: ')
      print(classification_report(self.y_test, y_pred), '\n')
      print('Confusion Matrix: ')
      print(confusion_matrix(self.y_test, y_pred), '\n')
      print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

      self.classification_report_plot(classification_report(self.y_test, y_pred, \
                                                            output_dict=True), "RF")

      if len(self.X_train[0]) == 2:
          self.classification_view(self.X_train, self.y_train, ANN_Classifier)

    def SVM(self, kernel_type):
        from sklearn.svm import SVC
        SVM_Classifier = SVC(kernel=kernel_type)
        SVM_Classifier.fit(self.X_train, self.y_train)
        y_pred = SVM_Classifier.predict(self.X_test)

        print("\n")
        print("*************************Support Vector Classifier************************* \n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        self.classification_report_plot(classification_report(self.y_test, y_pred, \
                                                              output_dict=True), "SVC" + kernel_type)

        if len(self.X_train[0]) == 2:
            self.classification_view(self.X_train, self.y_train, SVM_Classifier)