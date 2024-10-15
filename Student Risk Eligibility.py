#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


#importing the libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from tabulate import tabulate


# # Analysis on clean dataset

# In[2]:


#loading the clean dataset

student_clean_data = pd.read_csv('D:/MPS - Analytics/ALY 6080 - Integrated Exp Learning/STUDENT RISK ANALYSIS - FEATURE DROPPING ANALYSIS/BTECH CGPA Feature Dropping/Student Risk Analysis_CleanedData.csv')


# In[3]:


print("Total  number of Rows and Columns:")
print(student_clean_data.shape)
print("Column Names:")
print(student_clean_data.columns)


# # Label Encoding

# In[4]:


#labeling the eligbile/not eligible column using Label Encoder

labelencoder = LabelEncoder()

student_clean_data['Eligibity Labels']  = labelencoder.fit_transform(student_clean_data["ELIGIBLE/NOT ELIGIBLE"])


# # Label Encoding for categorical parameters

# In[5]:


#labeling the columns using Label Encoder

labelencoder = LabelEncoder()

student_clean_data['Gender Labels']  = labelencoder.fit_transform(student_clean_data["GENDER "])
student_clean_data['Branch Labels']  = labelencoder.fit_transform(student_clean_data["BRANCH "])
student_clean_data['Internship Labels']  = labelencoder.fit_transform(student_clean_data["INTERNSHIP"])
student_clean_data['Category Labels']  = labelencoder.fit_transform(student_clean_data["CATEGORY "])

student_clean_data.head(5)


# # Correlation Plot after label encoding

# In[5]:


#plotting correlation matrix for checking correlation between variables and the eligibility labels

plt.figure(figsize = (20,9))
ax = plt.subplot()
sns.heatmap(student_clean_data.corr(),annot=True, fmt='.1f', ax=ax, cmap="YlGnBu")
ax.set_title('Correlation Plot')


# In[6]:


#new dataframe for training model

new_student_clean_data = pd.DataFrame()

new_student_clean_data['10TH CGPA'] = student_clean_data['10TH CGPA ']
new_student_clean_data['12TH  / DIP %'] = student_clean_data['12TH  / DIP % ']
#new_student_clean_data['B.TECH  CGPA'] = student_clean_data['B.TECH  CGPA ']
new_student_clean_data['NO OF ACTIVE BACKLOGS'] = student_clean_data['NO OF ACTIVE BACKLOGS ']
new_student_clean_data['NO OF PASSIVE BACKLOGS'] = student_clean_data['NO OF PASSIVE BACKLOGS ']
new_student_clean_data['Gender Labels'] = student_clean_data['Gender Labels']
new_student_clean_data['Internship Labels'] = student_clean_data['Internship Labels']
new_student_clean_data['Branch Labels'] = student_clean_data['Branch Labels']
new_student_clean_data['Category Labels'] = student_clean_data['Category Labels']


# In[7]:


#extracting features and labels from the dataset

X = new_student_clean_data
Y = student_clean_data['Eligibity Labels']


# In[8]:


#splitting data into training and testing set

X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=42, test_size=0.3)
print(X_train.shape)
print(X_test.shape)


# # Logistic Regression Model

# In[9]:


#logistic regression model

logisticregression_model = LogisticRegression()
logisticregression_model.fit(X_train, y_train)


# In[10]:


#prediction result of the model

predicted_LR = logisticregression_model.predict(X_test)
print("Predicted Labels:\n")
predicted_LR


# In[11]:


#accuracy of the logistic regression model for training and testing set

print('Accuracy of Logistic Regression on training set: {:.2f}'.format(logisticregression_model.score(X_train, y_train)))
print('Accuracy of Logistic Regression on test set:     {:.2f}'.format(logisticregression_model.score(X_test, y_test)))

testaccuracy_LR = logisticregression_model.score(X_test, y_test)
accuracy_result_LR = round(testaccuracy_LR,2)
accuracy_result_LR


# In[12]:


#confusion matrix for LR Model

confusion_matrix_LR = confusion_matrix(y_test, predicted_LR)


fig = sns.heatmap(confusion_matrix_LR, annot=True,  annot_kws={"size": 18}, cmap = 'PuRd',fmt='g')
fig.xaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.yaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.set_xlabel('Predicted Values')
fig.set_ylabel('Actual Values ');
fig.set_title('Confusion Matrix for LR Model')
sns.set(font_scale=1.4)


# In[13]:


#classification report for LR Model

print("\n Classification report %s:\n%s\n" % (logisticregression_model, metrics.classification_report(y_test, predicted_LR)))


# # Random Forest Classifier

# In[12]:


#random forest classifier


randomforest_model = RandomForestClassifier()
randomforest_model.fit(X_train, y_train)


# In[13]:


#prediction result of the model

predicted_RFC = randomforest_model.predict(X_test)
print("Predicted Labels:\n")
predicted_RFC


# In[30]:


#accuracy of the random forest classifier for training and testing set

print('Accuracy of Random Forest Classifier on training set: {:.2f}'.format(randomforest_model.score(X_train, y_train)))
print('Accuracy of Random Forest Classifier on test set:     {:.2f}'.format(randomforest_model.score(X_test, y_test)))

accuracy_result_RFC = randomforest_model.score(X_test, y_test)
accuracy_result_RFC


# In[31]:


#confusion matrix

confusion_matrix_RFC = confusion_matrix(y_test, predicted_RFC)


fig = sns.heatmap(confusion_matrix_RFC, annot=True,  annot_kws={"size": 18}, cmap = 'PuRd',fmt='g')
fig.xaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.yaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.set_xlabel('Predicted Values')
fig.set_ylabel('Actual Values ');
fig.set_title('Confusion Matrix for Random Forest Classifier')
sns.set(font_scale=1.4)


# In[32]:


#classification report for Random Forest Classifier Model

print("\n Classification report %s:\n%s\n" % (randomforest_model, metrics.classification_report(y_test, predicted_RFC)))


# In[19]:


#feature importance

importance = randomforest_model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# # Feature Importance Visual for Random Forest Classifier

# In[14]:


(pd.Series(randomforest_model.feature_importances_, index=X.columns)
   .nlargest(8)
   .plot(kind='barh'))


# # SVM Linear Kernel

# In[20]:


#svm classifier

svm_model = svm.SVC(kernel='linear') # Linear Kernel
svm_model.fit(X_train, y_train)


# In[21]:


#predict the response for test dataset

predicted_SVM = svm_model.predict(X_test)
print("Predicted Labels:\n")
predicted_SVM


# In[22]:


#accuracy of the SVM model for training and testing set

print('Accuracy of SVM on training set: {:.2f}'.format(svm_model.score(X_train, y_train)))
print('Accuracy of SVM on test set:     {:.2f}'.format(svm_model.score(X_test, y_test)))

accuracy_result_SVM = svm_model.score(X_test, y_test)
accuracy_result_SVM = round(accuracy_result_SVM,3)
accuracy_result_SVM


# In[23]:


#confusion matrix

confusion_matrix_SVM = confusion_matrix(y_test, predicted_SVM)


fig = sns.heatmap(confusion_matrix_SVM, annot=True,  annot_kws={"size": 18}, cmap = 'PuRd',fmt='g')
fig.xaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.yaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.set_xlabel('Predicted Values')
fig.set_ylabel('Actual Values ');
fig.set_title('Confusion Matrix for SVM Linear Kernel')
sns.set(font_scale=1.4)


# In[24]:


#classification report for SVM Model

print("\n Classification report %s:\n%s\n" % (svm_model, metrics.classification_report(y_test, predicted_SVM)))


# # XGBoost Classifier

# In[25]:


#xgboost classifier

model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)


# In[26]:


#predict the response for test dataset

predicted_XGB = model_XGB.predict(X_test)
print("Predicted Labels:\n")
predicted_XGB


# In[27]:


#accuracy of the XGB model for training and testing set

print('Accuracy of XGB on training set: {:.2f}'.format(model_XGB.score(X_train, y_train)))
print('Accuracy of XGB on test set:     {:.2f}'.format(model_XGB.score(X_test, y_test)))

accuracy_result_XGB = model_XGB.score(X_test, y_test)
accuracy_result_XGB = round(accuracy_result_XGB,3)
accuracy_result_XGB


# In[28]:


#confusion matrix

confusion_matrix_XGB = confusion_matrix(y_test, predicted_XGB)


fig = sns.heatmap(confusion_matrix_XGB, annot=True,  annot_kws={"size": 18}, cmap = 'PuRd',fmt='g')
fig.xaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.yaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.set_xlabel('Predicted Values')
fig.set_ylabel('Actual Values ');
fig.set_title('Confusion Matrix for XGBoost Classifier')
sns.set(font_scale=1.4)


# In[29]:


#classification report for XGB Model

print("\n Classification report %s:\n%s\n" % (model_XGB, metrics.classification_report(y_test, predicted_XGB)))


# # Decision Tree Classifier

# In[30]:


#decision tree classifier

model_decisiontree = DecisionTreeClassifier()
model_decisiontree.fit(X_train, y_train)


# In[31]:


#predict the response for test dataset

predicted_decisiontree = model_decisiontree.predict(X_test)
print("Predicted Labels:\n")
predicted_decisiontree


# In[32]:


#accuracy of the Decision Tree model for training and testing set

print('Accuracy of Decision Tree Classifier on training set: {:.2f}'.format(model_decisiontree.score(X_train, y_train)))
print('Accuracy of Decision Tree Classifier on test set:     {:.2f}'.format(model_decisiontree.score(X_test, y_test)))

accuracy_result_decisiontree = model_decisiontree.score(X_test, y_test)
accuracy_result_decisiontree = round(accuracy_result_decisiontree,3)
accuracy_result_decisiontree


# In[33]:


#confusion matrix

confusion_matrix_decisiontree = confusion_matrix(y_test, predicted_decisiontree)


fig = sns.heatmap(confusion_matrix_decisiontree, annot=True,  annot_kws={"size": 18}, cmap = 'PuRd',fmt='g')
fig.xaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.yaxis.set_ticklabels(['Eligible','Not Eligible'])
fig.set_xlabel('Predicted Values')
fig.set_ylabel('Actual Values ');
fig.set_title('Confusion Matrix for Decision Tree Classifier')
sns.set(font_scale=1.4)


# In[34]:


#classification report for Decision Tree Model

print("\n Classification report %s:\n%s\n" % (model_decisiontree, metrics.classification_report(y_test, predicted_decisiontree)))


# # Comparing the accuracy of the models

# In[35]:


#comparing accuracy of all models built

accuracy_data = [["Logistic Regression", accuracy_result_LR],["Random Forest Classifier", accuracy_result_RFC], ["SVM Linear Kernel", accuracy_result_SVM], ["XGBoost Classifier", accuracy_result_XGB], ['Decision Tree Classifier', accuracy_result_decisiontree]]


head = ["Machine Learning Model", "Accuracy of the model"]
print(tabulate(accuracy_data, headers=head))


# # Feature Importance - Logistic Regression Model

# In[15]:


importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': logisticregression_model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)


# In[23]:


plt.bar(height=importances['Attribute'], x=importances['Importance'], color='#087E8B')
plt.title('Feature Importance', size=15)
plt.xticks(rotation='horizontal')
plt.show()


# In[37]:


#pickling the model

import pickle
pickle_out = open("Student_Classifier.pkl", "wb")
pickle.dump(logisticregression_model, pickle_out)
pickle_out.close()

