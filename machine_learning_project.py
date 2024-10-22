'''
The project provides an indication of the likelihood of developing heart disease using analysis models such as Decision Tree,
Logistic Regression, Naïve Bayes, and SVM. Dimensionality reduction is performed using PCA,
and performance is evaluated through metrics such as Accuracy, Precision, Recall, ROC curve, and more.

Please make sure to adjust the path to the data files.
The full code is attached, along with an example run.
'''

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import plotly
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_table('/content/gdrive/MyDrive/פרויקט למידת מכונה/AMI_GSE66360_series_matrix.csv', sep=',', header=0)
df

df=df.loc[:,~df.columns.duplicated()].copy()#Downloads repeating patterns מוריד שורות שחוזרות על עצמן
df.dropna(axis=0, inplace=True)

df= df.set_index(df.loc[:,'Class'])#Put the names of the genes in the first column
df.drop(['Class'],axis=1,inplace=True)

df=df.T  #שחלוף טבלה
df

df

# Import labels (for the whole dataset, both training and testing)
y = pd.read_csv('/content/gdrive/MyDrive/פרויקט למידת מכונה/type.csv')
print(y.shape)
y.head()

#כאן ניתן לראות שיש חלוקה טובה
y['type'].value_counts()

#סידור האינדקסים
df = df.reset_index(drop=True)
y = y.reset_index(drop=True)

# Recode label to numeric
y = y.replace({'H':0,'M':1})
labels = ['H', 'M'] # for plotting convenience later on
y

column_names = list(df.columns.values)#שם ברשימה את כל שמות בגנים שבשורה 0

X=df[column_names] # Features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)#חלוקה לבדיקה ואימון
#בחלק זה ניסינו כמה אופציות של חלוקת הנתונים ולרנדומליות, לקבלת הביצועים הטובים ביותר

print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of testing examples: {X_test.shape[0]}")

X_train

X_test

y_train

y_test

def printResult(y_test, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("f1:", metrics.f1_score(y_test, y_pred))
    print("Roc:", metrics.roc_auc_score(y_test, y_pred))

from sklearn.decomposition import PCA
pca = PCA()
X_train_pca1=pca.fit_transform(X_train)
print(type(X_train_pca1))

total = sum(pca.explained_variance_)

k = 0
current_variance = 0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    #print(pca.explained_variance_[k])
    k = k + 1

print(k, " features explain around 90% of the variance. From 54628 features to ", k, ", not too bad.", sep='')

pca = PCA(n_components=k)
X_train.pca = pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

var_exp = pca.explained_variance_ratio_.cumsum()
var_exp = var_exp*100
plt.bar(range(k), var_exp);


'''
pca3 = PCA(n_components=3).fit(X_train)
X_train_reduced = pca3.transform(X_train)

plt.clf()
fig = plt.figure(1, figsize=(10,6 ))
ax = Axes3D(fig, elev=-150, azim=110,)
#print(X_train_reduced[:, 0])
ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], X_train_reduced[:, 2], c = y_train.iloc[:,1], cmap = plt.cm.Paired, linewidths=10)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
print(fig)
'''

# Apply the same scaling to both datasets
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test) # note that we transform rather than fit_transform

# ננסה גישת אשכולות ללא פיקוח תוך שימוש בנתונים המותאמים
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_scl)
km_pred = kmeans.predict(X_test_scl)

print('K-means accuracy:', round(accuracy_score(y_test.iloc[:,1], km_pred), 3))

cm_km = confusion_matrix(y_test.iloc[:,1], km_pred)

ax = plt.subplot()
sns.heatmap(cm_km, annot=True, ax = ax, fmt='g', cmap='Greens')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('K-means Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels, rotation=360);

#Create a Gaussian classifier
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.iloc[:,1])

nb_pred = nb_model.predict(X_test)

print('Naive Bayes accuracy:', round(accuracy_score(y_test.iloc[:,1], nb_pred), 3))

cm_nb =  confusion_matrix(y_test.iloc[:,1], nb_pred)

ax = plt.subplot()
sns.heatmap(cm_nb, annot=True, ax = ax, fmt='g', cmap='Greens')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Naive Bayes Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels, rotation=360);

log_grid = {'C': [1e-03, 1e-2, 1e-1, 1, 10],
                 'penalty': ['l1', 'l2']}

log_estimator = LogisticRegression(solver='liblinear')

log_model = GridSearchCV(estimator=log_estimator,
                  param_grid=log_grid,
                  cv=3,
                  scoring='accuracy')

log_model.fit(X_train, y_train.iloc[:,1])

print("Best Parameters:\n", log_model.best_params_)

# Select best log model
best_log = log_model.best_estimator_

# Make predictions using the optimised parameters
log_pred = best_log.predict(X_test)

print('Logistic Regression accuracy:', round(accuracy_score(y_test.iloc[:,1], log_pred), 3))

cm_log =  confusion_matrix(y_test.iloc[:,1], log_pred)

ax = plt.subplot()
sns.heatmap(cm_log, annot=True, ax = ax, fmt='g', cmap='Greens')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Logistic Regression Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels, rotation=360);

# Parameter grid
svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10], "kernel": ["linear", "rbf", "poly"], "decision_function_shape" : ["ovo", "ovr"]}

# Create SVM grid search classifier
svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=3)

# Train the classifier
svm_grid.fit(X_train_pca, y_train.iloc[:,1])

print("Best Parameters:\n", svm_grid.best_params_)

# Select best svc
best_svc = svm_grid.best_estimator_

# Make predictions using the optimised parameters
svm_pred = best_svc.predict(X_test_pca)

print('SVM accuracy:', round(accuracy_score(y_test.iloc[:,1], svm_pred), 3))

cm_svm =  confusion_matrix(y_test.iloc[:,1], svm_pred)

ax = plt.subplot()
sns.heatmap(cm_svm, annot=True, ax = ax, fmt='g', cmap='Greens')

# Labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('SVM Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels, rotation=360);

#Logistic Regression
def logist(X_train, X_test, y_train):
    # instantiate the model
    logreg = LogisticRegression(solver='liblinear')
    # fit the model with data
    logreg.fit(X_train, y_train)
    # predicting
    y_pred= logreg.predict(X_test)
    return y_pred

#Decision Tree
def j48(X_train, X_test, y_train):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    return y_pred

#naivBase
def naivBase(X_train, X_test, y_train):
    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    return y_pred

#svm
def svm(X_train, X_test, y_train):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return y_pred

#פונקציית עזר לציור הגרפים
def graf(data, nameY):
    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color='maroon',
            width=0.4)

    plt.xlabel("classify")
    plt.ylabel(nameY)
    plt.title("result classify of kind "+ nameY)
    plt.show()

y_train_list=y_train.type
y_test_list=y_test.type

#שליחות לפונקציות המודלים
y_pred_log = logist(X_train, X_test, y_train_list)
y_pred_j48 = j48(X_train, X_test, y_train_list)
y_pred_naivBase = naivBase(X_train, X_test, y_train_list)
y_pred_svm = svm(X_train, X_test, y_train_list)

#שליחות לפונקציות המודלים
#_pca
y_pred_log_pca = logist(X_train_pca, X_test_pca, y_train_list)
y_pred_j48_pca = j48(X_train_pca, X_test_pca, y_train_list)
y_pred_naivBase_pca = naivBase(X_train_pca, X_test_pca, y_train_list)
y_pred_svm_pca= svm(X_train_pca, X_test_pca, y_train_list)

#  float עבור מודלים אלו נדרש דטה מסוג

y_pred_j48_f = y_pred_j48.astype(float)
y_test_list_f=y_test_list.astype(float)

#  float עבור מודלים אלו נדרש דטה מסוג
#_pca
y_pred_j48_f_pca = y_pred_j48_pca.astype(float)
y_test_list_f=y_test_list.astype(float)

# creating the dataset
#יוצרים את הדטה המתאים עבור כל מודל ומצריירים בגרף עבור כל מדידה
#_pca
data_Accuracy = {'Logistic_Regression': metrics.accuracy_score(y_test_list, y_pred_log_pca),
        'Naive_Bayes': metrics.accuracy_score(y_test_list, y_pred_naivBase_pca),
        'Decision_Tree': metrics.accuracy_score(y_test_list, y_pred_j48_pca),
        'SVM': metrics.accuracy_score(y_test_list, y_pred_svm_pca)}

data_precision = {'Logistic_Regression': metrics.precision_score(y_test_list, y_pred_log_pca),
        'Naive_Bayes': metrics.precision_score(y_test_list, y_pred_naivBase_pca),
        'Decision_Tree': metrics.precision_score(y_test_list, y_pred_j48_pca),
        'SVM': metrics.precision_score(y_test_list, y_pred_svm_pca)}

data_recall = {'Logistic_Regression': metrics.recall_score(y_test_list, y_pred_log_pca),
       'Naive_Bayes': metrics.recall_score(y_test_list, y_pred_naivBase_pca),
       'Decision_Tree': metrics.recall_score(y_test_list_f, y_pred_j48_f_pca),
       'SVM': metrics.recall_score(y_test_list, y_pred_svm_pca)}

data_f1 = {'Logistic_Regression': metrics.f1_score(y_test_list, y_pred_log_pca),
                     'Naive_Bayes': metrics.f1_score(y_test_list, y_pred_naivBase_pca),
                     'Decision_Tree': metrics.f1_score(y_test_list, y_pred_j48_pca),
                     'SVM': metrics.f1_score(y_test_list, y_pred_svm_pca)}

data_roc = {'Logistic_Regression': metrics.roc_auc_score(y_test_list, y_pred_log_pca),
                     'Naive_Bayes': metrics.roc_auc_score(y_test_list, y_pred_naivBase_pca),
                     'Decision_Tree': metrics.roc_auc_score(y_test_list, y_pred_j48_pca),
                     'SVM': metrics.roc_auc_score(y_test_list, y_pred_svm_pca)}

graf(data_Accuracy, "Accuracy")
graf(data_precision, "Precision")
graf(data_recall, "Recall")
graf(data_f1, "f1")
graf(data_roc, "Roc")

print(data_Accuracy)
print(data_precision)
print(data_recall)
print(data_f1)
print(data_roc)

# creating the dataset
data_Accuracy = {'Logistic_Regression': metrics.accuracy_score(y_test_list, y_pred_log),
        'Naive_Bayes': metrics.accuracy_score(y_test_list, y_pred_naivBase),
        'Decision_Tree': metrics.accuracy_score(y_test_list, y_pred_j48),
        'SVM': metrics.accuracy_score(y_test_list, y_pred_svm)}

data_precision = {'Logistic_Regression': metrics.precision_score(y_test_list, y_pred_log),
        'Naive_Bayes': metrics.precision_score(y_test_list, y_pred_naivBase),
        'Decision_Tree': metrics.precision_score(y_test_list, y_pred_j48),
        'SVM': metrics.precision_score(y_test_list, y_pred_svm)}

data_recall = {'Logistic_Regression': metrics.recall_score(y_test_list, y_pred_log),
       'Decision_Tree': metrics.recall_score(y_test_list_f, y_pred_j48_f),
       'Naive_Bayes': metrics.recall_score(y_test_list, y_pred_naivBase),
       'SVM': metrics.recall_score(y_test_list, y_pred_svm)}

data_f1 = {'Logistic_Regression': metrics.f1_score(y_test_list, y_pred_log),
                     'Naive_Bayes': metrics.f1_score(y_test_list, y_pred_naivBase),
                     'Decision_Tree': metrics.f1_score(y_test_list, y_pred_j48),
                     'SVM': metrics.f1_score(y_test_list, y_pred_svm)}

data_roc = {'Logistic_Regression': metrics.roc_auc_score(y_test_list, y_pred_log),
                     'Naive_Bayes': metrics.roc_auc_score(y_test_list, y_pred_naivBase),
                     'Decision_Tree': metrics.roc_auc_score(y_test_list, y_pred_j48),
                     'SVM': metrics.roc_auc_score(y_test_list, y_pred_svm)}



graf(data_Accuracy, "Accuracy")
graf(data_precision, "Precision")
graf(data_recall, "Recall")
graf(data_f1, "f1")
graf(data_roc, "Roc")

print(data_Accuracy)
print(data_precision)
print(data_recall)
print(data_f1)
print(data_roc)