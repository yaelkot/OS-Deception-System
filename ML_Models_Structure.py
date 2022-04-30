import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
import joblib
from os import listdir
from os.path import isfile
from sklearn.model_selection import KFold, cross_val_score


class ClassifierModel:

    def __init__(self, dataset, x_iloc_list, y_iloc, testSize):

        # From dataset:
        X = dataset.iloc[:, x_iloc_list].values
        y = dataset.iloc[:, y_iloc].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=0)

        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        self.X = X
        self.Y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_accuracy = []

    # ****************** Scores: ************************************

    def accuracy(self, confusion_matrix):
        sum, total = 0, 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                if i == j:
                    sum += confusion_matrix[i, j]
                total += confusion_matrix[i, j]
        return sum/total

    def classification_report_plot(self, clf_report, filename):
        folder = "clf_plots_monday"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        out_file_name = folder + "/" + filename + ".png"

        fig = plt.figure(figsize=(16, 10))
        sns.set(font_scale=4)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")
        fig.savefig(out_file_name, bbox_inches="tight")

    def k_fold(self, estimator, k, estimator_name):

        kfold = KFold(n_splits=k, shuffle=True, random_state=np.random.seed(7))
        results = cross_val_score(estimator, self.X, self.Y, cv=kfold)
        print(f"****{estimator_name}:****")
        self.models_accuracy.append((results.mean(), results.std()))
        print("Baseline accuracy: (%.2f%%) with std: (%.2f%%)" % (results.mean()*100, results.std()*100))

    # ****************** MODELS: ************************************

    def ANN(self):

        ANN_Classifier = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(7, 5), random_state=1)
        ANN_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(ANN_Classifier, "model/ann.sav")
        y_pred = ANN_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Nueral Network Classifier ************************* \n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')
        num_of_folds = 5
        self.k_fold(ANN_Classifier, num_of_folds, "ANN")

        self.classification_report_plot(classification_report(self.y_test, y_pred, output_dict=True), "ANN")

    def SVM(self, kernel_type = "linear"):
        SVM_Classifier = SVC()
        SVM_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(SVM_Classifier, "model/svm" + kernel_type + '.sav')
        y_pred = SVM_Classifier.predict(self.X_test)

        print("\n")
        print("*************************Support Vector Classifier************************* \n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        num_of_folds = 5
        self.k_fold(SVM_Classifier, num_of_folds, "SVM")

        self.classification_report_plot(classification_report(self.y_test, y_pred, output_dict=True), "SVM" + kernel_type)

    def RF(self):
        RF_Classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
        RF_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(RF_Classifier, "model/rf.sav")
        y_pred = RF_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Random Forest Classifier ************************* \n")
        print('Classification Report: ')
        p = classification_report(self.y_test, y_pred)
        print(p, '\n')
        # print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        num_of_folds = 5
        self.k_fold(RF_Classifier, num_of_folds, "RF")

        self.classification_report_plot(classification_report(self.y_test, y_pred, output_dict=True), "RF")

    def NB(self):
        NB_Classifier = GaussianNB()
        NB_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(NB_Classifier, "model/nb.sav")
        y_pred = NB_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Naive Bayes Classifier *************************\n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        num_of_folds = 5
        self.k_fold(NB_Classifier, num_of_folds, "NB")

        self.classification_report_plot(classification_report(self.y_test, y_pred, output_dict=True), "NB")

    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        KNN_Classifier = KNeighborsClassifier()
        KNN_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(KNN_Classifier, "model/knn.sav")
        y_pred = KNN_Classifier.predict(self.X_test)

        print("\n")
        print("************************* K-Neighbors Classifier *************************\n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        num_of_folds = 5
        self.k_fold(KNN_Classifier, num_of_folds, "KNN")

        self.classification_report_plot(classification_report(self.y_test, y_pred,output_dict=True), "KNN")

    def DT(self):
        DT_Classifier = DecisionTreeClassifier()
        DT_Classifier.fit(self.X_train, self.y_train)
        joblib.dump(DT_Classifier, "model/dt.sav")
        y_pred = DT_Classifier.predict(self.X_test)

        print("\n")
        print("************************* Decision Tree Classifier *************************\n")
        print('Classification Report: ')
        print(classification_report(self.y_test, y_pred), '\n')
        print('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred)) * 100, '%')

        num_of_folds = 5
        self.k_fold(DT_Classifier, num_of_folds, "DT")

        self.classification_report_plot(classification_report(self.y_test, y_pred,output_dict=True), "DT")

    def models_summery(self):
        folder = "clf_plots_monday"
        if not os.path.isdir(folder):
            os.mkdir(folder)
        out_file_name = folder + "/summary.png"
        accuracies = pd.DataFrame(
            self.models_accuracy, columns=['Accuracy', 'Std'],
            index=['KNN', 'linearSVM', 'rbfSVM', 'NB', 'RF', 'ANN', 'DT'])
        fig = plt.figure(figsize=(16, 10))
        sns.set(font_scale=4)
        sns.heatmap(accuracies, annot=True, cmap="BuPu")
        fig.savefig(out_file_name, bbox_inches="tight")


    def run_models(self, dataset, x_iloc_list, os_name):

        X = dataset.iloc[:, x_iloc_list].values
        Y = [os_name for _ in range(dataset.shape[0])]
        models = ["model/" + f for f in listdir("model") if isfile("model/" + f)]
        for filename in models:
            print(filename)
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X, Y)
            print(result)
            pred_cols = pd.Series(loaded_model.predict(X))
            print(set(pred_cols))
            print("******************\n")



