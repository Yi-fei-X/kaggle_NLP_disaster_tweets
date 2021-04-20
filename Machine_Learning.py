import numpy as np
import pandas as pd
import random
import re
import csv
import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from prettytable import PrettyTable
random.seed(0)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

#Read the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_keywords = train_df['keyword']

#Clean the data
def cleandata(text):
    text = text.lower()
    #Remove webpage
    text = re.sub(r'http://\S+', '', text)
    text = re.sub(r'https://\S+', '', text)
    text = re.sub(r'http', '', text)
    #Remove mention
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'@', '', text)
    text = re.sub(r'via', '', text)
    #Remove some sign
    text = re.sub(r'#', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'[*]', '', text)
    text = re.sub(r';\)', '', text)
    text = re.sub(r':\)', '', text)
    text = re.sub(r'-', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r';', '', text)
    text = re.sub(r'<', '', text)
    text = re.sub(r'=', '', text)
    text = re.sub(r'>', '', text)
    text = re.sub('\+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\|', '', text)
    text = re.sub('\[', '', text)
    text = re.sub('\]', '', text)
    text = re.sub('\(', '', text)
    text = re.sub('\)', '', text)
    #Remove redundant sign
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub('\s+', ' ', text).strip()
    #Remove non-ascii
    text = text.encode("ascii", errors="ignore").decode()
    return text

train_df['text'] = train_df['text'].apply(cleandata)
test_df['text'] = test_df['text'].apply(cleandata)

#Compute TFIDF
tfidf = TfidfVectorizer(min_df=3)
train_tfidf = tfidf.fit_transform(train_df["text"])
test_tfidf = tfidf.transform(test_df["text"])

#Model
#Random Forest
def Random_forest (train_val_X, train_val_y):
    num_trees = np.arange(50,200,20) #np.arange(10,300,10)
    train_acc_forest = []
    val_acc_forest = []

    for num_tree in num_trees:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            dtc = RandomForestClassifier(n_estimators=num_tree)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        print("Number of trees", num_tree)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_forest.append(avg_train_acc)
        val_acc_forest.append(avg_val_acc)

    return train_acc_forest, val_acc_forest, num_trees

train_acc_forest, val_acc_forest, num_trees = Random_forest(train_tfidf, train_df['target'])
#Form a table
num_trees = num_trees.tolist()
Table_Q2_forest = PrettyTable()
Table_Q2_forest_title = num_trees.copy()
Table_Q2_forest_title.insert(0,'type/number of trees')
Table_Q2_forest.field_names = Table_Q2_forest_title
Table_Q2_train_acc_forest = train_acc_forest.copy()
Table_Q2_train_acc_forest.insert(0,"train_acc_forest")
Table_Q2_forest.add_row(Table_Q2_train_acc_forest)
Table_Q2_val_acc_forest = val_acc_forest.copy()
Table_Q2_val_acc_forest.insert(0,"val_acc_forest")
Table_Q2_forest.add_row(Table_Q2_val_acc_forest)
print(Table_Q2_forest)

#XGBOOST
def XGBOOST(train_val_X, train_val_y):
    etas = np.arange(0.1,1,0.2) #np.arange(0.1,1,0.2)
    train_acc_xgboost = []
    val_acc_xgboost = []

    for eta in etas:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            dtc = xgb.XGBClassifier(eta=eta)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        print("Number of eta", eta)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_xgboost.append(avg_train_acc)
        val_acc_xgboost.append(avg_val_acc)

    return train_acc_xgboost, val_acc_xgboost, etas
train_acc_xgboost, val_acc_xgboost, etas = XGBOOST(train_tfidf, train_df['target'])

#Form a table
etas = etas.tolist()
Table_Q2_xgboost = PrettyTable()
Table_Q2_xgboost_title = etas.copy()
Table_Q2_xgboost_title.insert(0,'type/etas')
Table_Q2_xgboost.field_names = Table_Q2_xgboost_title
Table_Q2_train_acc_xgboost = train_acc_xgboost.copy()
Table_Q2_train_acc_xgboost.insert(0,"train_acc_xgboost")
Table_Q2_xgboost.add_row(Table_Q2_train_acc_xgboost)
Table_Q2_val_acc_xgboost = val_acc_xgboost.copy()
Table_Q2_val_acc_xgboost.insert(0,"val_acc_xgboost")
Table_Q2_xgboost.add_row(Table_Q2_val_acc_xgboost)
print(Table_Q2_xgboost)

#SVM
def SVM(train_val_X, train_val_y):
    C_svms = np.arange(0.1,1,0.2) #np.arange(0.1,1,0.2)
    train_acc_SVM = []
    val_acc_SVM = []

    for C_svm in C_svms:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            SVM = svm.SVC(C=C_svm)
            SVM.fit(train_X, train_y)
            train_acc.append(SVM.score(train_X, train_y))
            val_acc.append(SVM.score(val_X, val_y))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        print("Number of C", C_svm)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_SVM.append(avg_train_acc)
        val_acc_SVM.append(avg_val_acc)

    return train_acc_SVM, val_acc_SVM, C_svms
train_acc_SVM, val_acc_SVM, C_svms = SVM(train_tfidf, train_df['target'])

#Form a table
C_svms = C_svms.tolist()
Table_Q2_SVM = PrettyTable()
Table_Q2_SVM_title = C_svms.copy()
Table_Q2_SVM_title.insert(0,'type/C')
Table_Q2_SVM.field_names = Table_Q2_SVM_title
Table_Q2_train_acc_SVM = train_acc_SVM.copy()
Table_Q2_train_acc_SVM.insert(0,"train_acc_SVM")
Table_Q2_SVM.add_row(Table_Q2_train_acc_SVM)
Table_Q2_val_acc_SVM = val_acc_SVM.copy()
Table_Q2_val_acc_SVM.insert(0,"val_acc_SVM")
Table_Q2_SVM.add_row(Table_Q2_val_acc_SVM)
print(Table_Q2_SVM)

#Prediction, based on the best model SVM
SVM = svm.SVC(C=0.9)
SVM.fit(train_tfidf, train_df['target'])
train_acc = SVM.score(train_tfidf, train_df['target'])
test_pred = SVM.predict(test_tfidf)

#Form a table
Table_best = PrettyTable()
Table_best_title = ['SVM']
Table_best_title.insert(0,'Best Model')
Table_best.field_names = Table_best_title
Table_best_acc = []
Table_best_acc.insert(0,"train_acc")
Table_best_acc.insert(1, train_acc)
Table_best.add_row(Table_best_acc)
print(Table_best)

len_test = len(test_pred)
index_test = []
for i in range(len_test):
    index_test.append(test_df['id'][i])

with open("labels.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'target'])
    for i in range(len_test):
        writer.writerow([index_test[i], test_pred[i]])

print()
