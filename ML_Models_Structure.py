import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
import numpy as np



class ClassifierModel:

    def __init__(self, dataset, x_iloc_list, y_iloc, testSize):

        # From dataset:
        X = dataset.iloc[:, x_iloc_list].values
        y = dataset.iloc[:, y_iloc].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # ****************** Scores: ************************************

    def accuracy(self, confusion_matrix):
        sum, total = 0, 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                if i == j:
                    sum += confusion_matrix[i, j]
                total += confusion_matrix[i, j]
        return sum/total


    # TODO: change the function
    def classification_report_plot(self, clf_report, filename):
        folder = "clf_plots_new"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        out_file_name = folder + "/" + filename + ".png"

        fig = plt.figure(figsize=(16, 10))
        sns.set(font_scale=4)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Reds")
        fig.savefig(out_file_name, bbox_inches="tight")




    # ****************** MODELS: ************************************

    def ANN(self):
        ANN_Classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        ANN_Classifier.fit(self.X_train,self.y_train)
        y_pred = ANN_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Nueral Network Classifier ************************* \n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        # print('Confusion Matrix: ')
        # print(confusion_matrix(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        # self.classification_report_plot(classification_report(self.y_test, y_pred), "ANN")

        # if len(self.X_train[0]) == 2:
        #     self.classification_view(self.X_train, self.y_train, ANN_Classifier)

    def SVM(self, kernel_type):
        SVM_Classifier = SVC(kernel=kernel_type)
        SVM_Classifier.fit(self.X_train, self.y_train)
        y_pred = SVM_Classifier.predict(self.X_test)

        print("\n")
        print("*************************Support Vector Classifier************************* \n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        # print('Confusion Matrix: ')
        # print(confusion_matrix(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        # self.classification_report_plot(classification_report(self.y_test, y_pred), "SVC" + kernel_type)

        # if len(self.X_train[0]) == 2:
        #     self.classification_view(self.X_train, self.y_train, SVM_Classifier)

    def RF(self):
        RF_Classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        RF_Classifier.fit(self.X_train, self.y_train)
        # joblib.dump(rf_classifier, "model/rf.sav")
        y_pred = RF_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Random Forest Classifier ************************* \n")
        print('Classification Report: ')
        p = classification_report(self.y_test, y_pred)
        print(p, '\n')
        # print('Confusion Matrix: ')
        # print(confusion_matrix(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')
        self.classification_report_plot(clf_report=classification_report(self.y_test, y_pred, output_dict=True),filename= "RF")

        # if len(self.X_train[0]) == 2:
        #     self.classification_view(self.X_train, self.y_train, RF_Classifier)