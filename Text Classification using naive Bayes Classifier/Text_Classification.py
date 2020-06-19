# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:30:15 2020

@author: win
"""
### Import Libraries ###
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

######### Loading Training Dataset into a dataframe ###########
data  = pd.read_csv(r'C:\Users\win\Downloads\AI Genie\AIGenie_Capstone2.0\sms.csv', encoding = 'latin1')

######### Loading Testing dataset into a dataframe ##############3
test_data = pd.read_csv(r'C:\Users\win\Downloads\AI Genie\AIGenie_Capstone2.0\Sms_unlabelled.csv', encoding = 'latin1')

################ Print 5 rows of Training dataframe ####################
data.head()

################ Print 5 rows of Test dataframe ####################
test_data.head()

############### Drop no. column ######################
data = data.drop(['no'], axis = 1)

###################### Checking fir null value if any ################333333
data.isna().values.sum()
test_data.isna().values.any()
################### Dropping NaN values #####################
data = data.dropna(axis =0)

#################### Counting for total no. of Ham and Spam #################3
data.result.value_counts()
data.groupby('result').describe()
data.describe()

################ Plotting graph for distribution ###############33
sns.countplot(x = 'result', data = data)
data.loc[:,'result'].value_counts()
plt.title('Distribution of Spam & Ham')

########################### Text Preprocessing #############################
def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
################### Pipeline #######################33333
pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])

#################### train test split ##############################33
msg_train,msg_test,label_train,label_test = train_test_split(data['sms'],data['result'],test_size=0.2)
print(len(msg_train),len(msg_test),len(label_train),len(label_test))

####################### Fitting the model on training data #####################
pipeline.fit(msg_train,label_train)

###################### Predicting on test data #####################3333
predictions = pipeline.predict(msg_test)

######################### Check for Accuracy Score #############################
print(classification_report(predictions,label_test))
print(confusion_matrix(label_test, predictions))
print(accuracy_score(label_test, predictions))

############## Predictions on Unlabelled dataset ########################
test_data['result'] = pipeline.predict(test_data['sms'])
test_data['result'].value_counts()
test_data = test_data.drop(['sms'],axis = 1)
test_data.to_csv(r'C:\Users\win\Downloads\AI Genie\AIGenie_Capstone2.0\Text_Classification_Output.csv',index = False)
