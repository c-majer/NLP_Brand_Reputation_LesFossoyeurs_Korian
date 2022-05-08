# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:06:00 2022

NLP Text Mining Sentiment Analysis for Brand Reputation Optimization

@author: camillo.majerczyk
"""
#%% Gen Comments
'''
# Original dataset columns:
     # Facility name
     # COMPLAINT_ID
     # TYPE
     # OPEN_DATE
     # COMMUNICATION_CHANNEL
     # REPORTED_TOPIC 1
     # COMPLIANT_DESCRIPTION
     # OPENING_CLASSIFICATION
'''

#%% IMPORTS

#from numpy import mean
#from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
#import sklearn
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import validation_curve
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#import pickle

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#import xlrd
from collections import defaultdict
#from nltk import ngrams

#%% Data preparation
Segn_data = pd.read_excel(r"C:\Users\camillo.majerczyk\Desktop\NLP_Segnalazioni\Dati_Segnalazioni_Genn2016_Marz2022.xls")
Segn_data.head()
Segn_data.rename(columns= {'REPORTED_TOPIC 1': "REPORTED_TOPIC_1"}, inplace = True)


#%% ------------------ FIRST RESEARCH QUESTION (1 -	What people complain about)

First_data = Segn_data.drop('OPENING_CLASSIFICATION', 1)
First_data = First_data.dropna()
First_data['TYPE'].value_counts()

# Remove punctuations
def remove_punct(text):
    for punct in string.punctuation:
        text = text.replace(punct, '')
    return text

First_data["COMPLIANT_DESCRIPTION"] = First_data["COMPLIANT_DESCRIPTION"].apply(remove_punct)

# Remove the stopwords 

stop_en = stopwords.words('english')
stop = stopwords.words('italian')
First_data['COMPLIANT_DESCRIPTION'] = First_data['COMPLIANT_DESCRIPTION'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))

# Function to separate n-grams 
def generate_N_grams(text,ngram=1):
  words=[word for word in text.split(" ")]  
  print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

First_data_complaints = First_data[First_data['TYPE'] == 'reclamo']
First_data_apprezza = First_data[First_data['TYPE'] == 'apprezzamento']

#%% ----- Unigram Reclami e Apprezzamenti

#get the count of every word
Reclami_values = defaultdict(int)
Apprezzamenti_values = defaultdict(int)

#count the reclami
for text in First_data_complaints.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text):
        Reclami_values[word]+=1

#count the apprezzamenti
for text in First_data_apprezza.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text):
        Apprezzamenti_values[word]+=1

#focus on more frequently occuring words for every sentiment
words_reclami  = pd.DataFrame(sorted(Reclami_values.items(), key=lambda x:x[1], reverse=True))
words_apprezza  = pd.DataFrame(sorted(Apprezzamenti_values.items(), key=lambda x:x[1], reverse=True))

words_reclami.to_excel('words_reclami.xlsx')
words_apprezza.to_excel('words_apprezza.xlsx')

first_fifty_reclami = words_reclami[:20]
first_fifty_apprezzamenti = words_apprezza[:20]

#plot first 20 reclami
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_fifty_reclami[0],first_fifty_reclami[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Reclami (Unigram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#plot first 20 apprezzamenti
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_fifty_apprezzamenti[0],first_fifty_apprezzamenti[1], color ='#e07204', width = 0.4)
plt.title("Top 20 words in Apprezzamenti (Unigram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#%% ----- Trigram Reclami e Apprezzamenti

Reclami_values_tren = defaultdict(int)
Apprezzamenti_values_tren = defaultdict(int)

#count the reclami
for text in First_data_complaints.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text, 3):
        Reclami_values_tren[word]+=1

#count the apprezzamenti
for text in First_data_apprezza.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text, 3):
        Apprezzamenti_values_tren[word]+=1

#focus on more frequently occuring words for every sentiment
words_reclami_tren  = pd.DataFrame(sorted(Reclami_values_tren.items(), key=lambda x:x[1], reverse=True))
words_apprezza_tren  = pd.DataFrame(sorted(Apprezzamenti_values_tren.items(), key=lambda x:x[1], reverse=True))

words_reclami_tren.to_excel('words_reclami_tren.xlsx')
words_apprezza_tren.to_excel('words_apprezza_tren.xlsx')

first_fifty_reclami_tren = words_reclami_tren[:20]
first_fifty_apprezzamenti_tren = words_apprezza_tren[:20]

#plot first 20 reclami
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_fifty_reclami_tren[0],first_fifty_reclami_tren[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Reclami (Trigram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#plot first 20 apprezzamenti
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_fifty_apprezzamenti_tren[0],first_fifty_apprezzamenti_tren[1], color ='#e07204', width = 0.4)
plt.title("Top 20 words in Apprezzamenti (Trigram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

## ----- 5-gram Complaints


Reclami_values_five = defaultdict(int)
Apprezzamenti_values_five = defaultdict(int)

#count the reclami
for text in First_data_complaints.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text, 5):
        Reclami_values_five[word]+=1

#count the apprezzamenti
for text in First_data_apprezza.COMPLIANT_DESCRIPTION:
    for word in generate_N_grams(text, 5):
        Apprezzamenti_values_five[word]+=1

#focus on more frequently occuring words for every sentiment
words_reclami_five  = pd.DataFrame(sorted(Reclami_values_five.items(), key=lambda x:x[1], reverse=True))
words_apprezza_five  = pd.DataFrame(sorted(Apprezzamenti_values_five.items(), key=lambda x:x[1], reverse=True))

words_reclami_five.to_excel('words_reclami_five.xlsx')
words_apprezza_five.to_excel('words_apprezza_five.xlsx')

first_fifty_reclami_five = words_reclami_five[:20]
first_fifty_apprezzamenti_five = words_apprezza_five[:20]

#plot first 20 reclami
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=90)
plt.bar(first_fifty_reclami_five[0],first_fifty_reclami_five[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Reclami (5-gram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#plot first 20 apprezzamenti
plt.figure(1,figsize=(46,12))
plt.xticks(rotation=90)
plt.bar(first_fifty_apprezzamenti_five[0],first_fifty_apprezzamenti_five[1], color ='#e07204', width = 0.4)
plt.title("Top 20 words in Apprezzamenti (5-gram)", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#%% LES FOSSOYEURS

LF_data = pd.read_excel(r"C:\Users\camillo.majerczyk\Desktop\NLP_Segnalazioni\LesFossoyeurs.xlsx")

stop_fr = stopwords.words('french')

# Remove punctuations
def remove_punct(text):
    for punct in string.punctuation+'»«':
        text = text.replace(punct, '')
    return text

#removing punctuation from chaps
LF_data['Intro'] = LF_data['Intro'].apply(remove_punct)
LF_data['Chap_1'] = LF_data['Chap_1'].apply(remove_punct)
LF_data['Chap_2'] = LF_data['Chap_2'].apply(remove_punct)
LF_data['Chap_3'] = LF_data['Chap_3'].apply(remove_punct)
LF_data['Chap_4'] = LF_data['Chap_4'].apply(remove_punct)
LF_data['Chap_5'] = LF_data['Chap_5'].apply(remove_punct)
LF_data['Chap_6'] = LF_data['Chap_6'].apply(remove_punct)
LF_data['Chap_7'] = LF_data['Chap_7'].apply(remove_punct)
LF_data['Chap_8'] = LF_data['Chap_8'].apply(remove_punct)
LF_data['Chap_9'] = LF_data['Chap_9'].apply(remove_punct)
LF_data['Chap_10'] = LF_data['Chap_10'].apply(remove_punct)
LF_data['Chap_11'] = LF_data['Chap_11'].apply(remove_punct)
LF_data['Chap_12'] = LF_data['Chap_12'].apply(remove_punct)
LF_data['Chap_13'] = LF_data['Chap_13'].apply(remove_punct)
LF_data['Chap_14'] = LF_data['Chap_14'].apply(remove_punct)
LF_data['Chap_15'] = LF_data['Chap_15'].apply(remove_punct)
LF_data['Chap_16'] = LF_data['Chap_16'].apply(remove_punct)
LF_data['Chap_17'] = LF_data['Chap_17'].apply(remove_punct)
LF_data['Chap_18'] = LF_data['Chap_18'].apply(remove_punct)
LF_data['Chap_19'] = LF_data['Chap_19'].apply(remove_punct)
LF_data['Chap_20'] = LF_data['Chap_20'].apply(remove_punct)
LF_data['Chap_21'] = LF_data['Chap_21'].apply(remove_punct)
LF_data['Chap_22'] = LF_data['Chap_22'].apply(remove_punct)
LF_data['Chap_23'] = LF_data['Chap_23'].apply(remove_punct)
LF_data['Chap_24'] = LF_data['Chap_24'].apply(remove_punct)
LF_data['Chap_25'] = LF_data['Chap_25'].apply(remove_punct)
LF_data['Chap_26'] = LF_data['Chap_26'].apply(remove_punct)
LF_data['Chap_27'] = LF_data['Chap_27'].apply(remove_punct)
LF_data['Chap_28'] = LF_data['Chap_28'].apply(remove_punct)
LF_data['Chap_29'] = LF_data['Chap_29'].apply(remove_punct)
LF_data['Chap_30'] = LF_data['Chap_30'].apply(remove_punct)
LF_data['Chap_31'] = LF_data['Chap_31'].apply(remove_punct)
LF_data['Chap_32'] = LF_data['Chap_32'].apply(remove_punct)
LF_data['Chap_33'] = LF_data['Chap_33'].apply(remove_punct)
LF_data['Chap_34'] = LF_data['Chap_34'].apply(remove_punct)
LF_data['Chap_35'] = LF_data['Chap_35'].apply(remove_punct)
LF_data['Chap_36'] = LF_data['Chap_36'].apply(remove_punct)
LF_data['Chap_37'] = LF_data['Chap_37'].apply(remove_punct)
LF_data['Chap_38'] = LF_data['Chap_38'].apply(remove_punct)

#removing stopwords from chaps
LF_data['Intro'] = LF_data['Intro'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_1'] = LF_data['Chap_1'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_2'] = LF_data['Chap_2'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_3'] = LF_data['Chap_3'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_4'] = LF_data['Chap_4'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_5'] = LF_data['Chap_5'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_6'] = LF_data['Chap_6'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_7'] = LF_data['Chap_7'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_8'] = LF_data['Chap_8'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_9'] = LF_data['Chap_9'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_10'] = LF_data['Chap_10'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_11'] = LF_data['Chap_11'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_12'] = LF_data['Chap_12'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_13'] = LF_data['Chap_13'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_14'] = LF_data['Chap_14'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_15'] = LF_data['Chap_15'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_16'] = LF_data['Chap_16'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_17'] = LF_data['Chap_17'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_18'] = LF_data['Chap_18'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_19'] = LF_data['Chap_19'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_20'] = LF_data['Chap_20'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_21'] = LF_data['Chap_21'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_22'] = LF_data['Chap_22'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_23'] = LF_data['Chap_23'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_24'] = LF_data['Chap_24'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_25'] = LF_data['Chap_25'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_26'] = LF_data['Chap_26'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_27'] = LF_data['Chap_27'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_28'] = LF_data['Chap_28'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_29'] = LF_data['Chap_29'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_30'] = LF_data['Chap_30'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_31'] = LF_data['Chap_31'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_32'] = LF_data['Chap_32'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_33'] = LF_data['Chap_33'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_34'] = LF_data['Chap_34'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_35'] = LF_data['Chap_35'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_36'] = LF_data['Chap_36'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_37'] = LF_data['Chap_37'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))
LF_data['Chap_38'] = LF_data['Chap_38'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_fr)]))

#Initialize empty dicts
Intro_values = defaultdict(int)
Chap_1_values = defaultdict(int)
Chap_2_values = defaultdict(int)
Chap_3_values = defaultdict(int)
Chap_4_values = defaultdict(int)
Chap_5_values = defaultdict(int)
Chap_6_values = defaultdict(int)
Chap_7_values = defaultdict(int)
Chap_8_values = defaultdict(int)
Chap_9_values = defaultdict(int)
Chap_10_values = defaultdict(int)
Chap_11_values = defaultdict(int)
Chap_12_values = defaultdict(int)
Chap_13_values = defaultdict(int)
Chap_14_values = defaultdict(int)
Chap_15_values = defaultdict(int)
Chap_16_values = defaultdict(int)
Chap_17_values = defaultdict(int)
Chap_18_values = defaultdict(int)
Chap_19_values = defaultdict(int)
Chap_20_values = defaultdict(int)
Chap_21_values = defaultdict(int)
Chap_22_values = defaultdict(int)
Chap_23_values = defaultdict(int)
Chap_24_values = defaultdict(int)
Chap_25_values = defaultdict(int)
Chap_26_values = defaultdict(int)
Chap_27_values = defaultdict(int)
Chap_28_values = defaultdict(int)
Chap_29_values = defaultdict(int)
Chap_30_values = defaultdict(int)
Chap_31_values = defaultdict(int)
Chap_32_values = defaultdict(int)
Chap_33_values = defaultdict(int)
Chap_34_values = defaultdict(int)
Chap_35_values = defaultdict(int)
Chap_36_values = defaultdict(int)
Chap_37_values = defaultdict(int)
Chap_38_values = defaultdict(int)

#counting per chapter
for text in LF_data.Intro:
    for word in generate_N_grams(text):
        Intro_values[word]+=1

for text in LF_data.Chap_1:
    for word in generate_N_grams(text):
        Chap_1_values[word]+=1

for text in LF_data.Chap_2:
    for word in generate_N_grams(text):
        Chap_2_values[word]+=1

for text in LF_data.Chap_3:
    for word in generate_N_grams(text):
        Chap_3_values[word]+=1

for text in LF_data.Chap_4:
    for word in generate_N_grams(text):
        Chap_4_values[word]+=1
        
for text in LF_data.Chap_5:
    for word in generate_N_grams(text):
        Chap_5_values[word]+=1

for text in LF_data.Chap_6:
    for word in generate_N_grams(text):
        Chap_6_values[word]+=1

for text in LF_data.Chap_7:
    for word in generate_N_grams(text):
        Chap_7_values[word]+=1
        
for text in LF_data.Chap_8:
    for word in generate_N_grams(text):
        Chap_8_values[word]+=1

for text in LF_data.Chap_9:
    for word in generate_N_grams(text):
        Chap_9_values[word]+=1

for text in LF_data.Chap_10:
    for word in generate_N_grams(text):
        Chap_10_values[word]+=1
        
for text in LF_data.Chap_11:
    for word in generate_N_grams(text):
        Chap_11_values[word]+=1

for text in LF_data.Chap_12:
    for word in generate_N_grams(text):
        Chap_12_values[word]+=1

for text in LF_data.Chap_13:
    for word in generate_N_grams(text):
        Chap_13_values[word]+=1
        
for text in LF_data.Chap_14:
    for word in generate_N_grams(text):
        Chap_14_values[word]+=1

for text in LF_data.Chap_15:
    for word in generate_N_grams(text):
        Chap_15_values[word]+=1

for text in LF_data.Chap_16:
    for word in generate_N_grams(text):
        Chap_16_values[word]+=1
        
for text in LF_data.Chap_17:
    for word in generate_N_grams(text):
        Chap_17_values[word]+=1

for text in LF_data.Chap_18:
    for word in generate_N_grams(text):
        Chap_18_values[word]+=1

for text in LF_data.Chap_19:
    for word in generate_N_grams(text):
        Chap_19_values[word]+=1

for text in LF_data.Chap_20:
    for word in generate_N_grams(text):
        Chap_20_values[word]+=1

for text in LF_data.Chap_21:
    for word in generate_N_grams(text):
        Chap_21_values[word]+=1

for text in LF_data.Chap_22:
    for word in generate_N_grams(text):
        Chap_22_values[word]+=1

for text in LF_data.Chap_23:
    for word in generate_N_grams(text):
        Chap_23_values[word]+=1

for text in LF_data.Chap_24:
    for word in generate_N_grams(text):
        Chap_24_values[word]+=1

for text in LF_data.Chap_25:
    for word in generate_N_grams(text):
        Chap_25_values[word]+=1
        
for text in LF_data.Chap_26:
    for word in generate_N_grams(text):
        Chap_26_values[word]+=1

for text in LF_data.Chap_27:
    for word in generate_N_grams(text):
        Chap_27_values[word]+=1

for text in LF_data.Chap_28:
    for word in generate_N_grams(text):
        Chap_28_values[word]+=1

for text in LF_data.Chap_29:
    for word in generate_N_grams(text):
        Chap_29_values[word]+=1

for text in LF_data.Chap_30:
    for word in generate_N_grams(text):
        Chap_30_values[word]+=1

for text in LF_data.Chap_31:
    for word in generate_N_grams(text):
        Chap_31_values[word]+=1

for text in LF_data.Chap_32:
    for word in generate_N_grams(text):
        Chap_32_values[word]+=1

for text in LF_data.Chap_33:
    for word in generate_N_grams(text):
        Chap_33_values[word]+=1

for text in LF_data.Chap_34:
    for word in generate_N_grams(text):
        Chap_34_values[word]+=1

for text in LF_data.Chap_35:
    for word in generate_N_grams(text):
        Chap_35_values[word]+=1

for text in LF_data.Chap_36:
    for word in generate_N_grams(text):
        Chap_36_values[word]+=1

for text in LF_data.Chap_37:
    for word in generate_N_grams(text):
        Chap_37_values[word]+=1

for text in LF_data.Chap_38:
    for word in generate_N_grams(text):
        Chap_38_values[word]+=1


#Sorting words per each chap
words_Intro  = pd.DataFrame(sorted(Intro_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_1  = pd.DataFrame(sorted(Chap_1_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_2  = pd.DataFrame(sorted(Chap_2_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_3  = pd.DataFrame(sorted(Chap_3_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_4  = pd.DataFrame(sorted(Chap_4_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_5  = pd.DataFrame(sorted(Chap_5_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_6  = pd.DataFrame(sorted(Chap_6_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_7  = pd.DataFrame(sorted(Chap_7_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_8  = pd.DataFrame(sorted(Chap_8_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_9  = pd.DataFrame(sorted(Chap_9_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_10  = pd.DataFrame(sorted(Chap_10_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_11  = pd.DataFrame(sorted(Chap_11_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_12  = pd.DataFrame(sorted(Chap_12_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_13  = pd.DataFrame(sorted(Chap_13_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_14  = pd.DataFrame(sorted(Chap_14_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_15  = pd.DataFrame(sorted(Chap_15_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_16  = pd.DataFrame(sorted(Chap_16_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_17  = pd.DataFrame(sorted(Chap_17_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_18  = pd.DataFrame(sorted(Chap_18_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_19  = pd.DataFrame(sorted(Chap_19_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_20  = pd.DataFrame(sorted(Chap_20_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_21  = pd.DataFrame(sorted(Chap_21_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_22  = pd.DataFrame(sorted(Chap_22_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_23  = pd.DataFrame(sorted(Chap_23_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_24  = pd.DataFrame(sorted(Chap_24_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_25  = pd.DataFrame(sorted(Chap_25_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_26  = pd.DataFrame(sorted(Chap_26_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_27  = pd.DataFrame(sorted(Chap_27_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_28  = pd.DataFrame(sorted(Chap_28_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_29  = pd.DataFrame(sorted(Chap_29_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_30  = pd.DataFrame(sorted(Chap_30_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_31  = pd.DataFrame(sorted(Chap_31_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_32  = pd.DataFrame(sorted(Chap_32_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_33  = pd.DataFrame(sorted(Chap_33_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_34  = pd.DataFrame(sorted(Chap_34_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_35  = pd.DataFrame(sorted(Chap_35_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_36  = pd.DataFrame(sorted(Chap_36_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_37  = pd.DataFrame(sorted(Chap_37_values.items(), key=lambda x:x[1], reverse=True))
words_Chap_38  = pd.DataFrame(sorted(Chap_38_values.items(), key=lambda x:x[1], reverse=True))

# Save the data

words_Intro.to_excel('words_Intro.xlsx')
words_Chap_1.to_excel('words_Chap_1.xlsx')
words_Chap_2.to_excel('words_Chap_2.xlsx')
words_Chap_3.to_excel('words_Chap_3.xlsx')
words_Chap_4.to_excel('words_Chap_4.xlsx')
words_Chap_5.to_excel('words_Chap_5.xlsx')
words_Chap_6.to_excel('words_Chap_6.xlsx')
words_Chap_7.to_excel('words_Chap_7.xlsx')
words_Chap_8.to_excel('words_Chap_8.xlsx')
words_Chap_9.to_excel('words_Chap_9.xlsx')
words_Chap_10.to_excel('words_Chap_10.xlsx')
words_Chap_11.to_excel('words_Chap_11.xlsx')
words_Chap_12.to_excel('words_Chap_12.xlsx')
words_Chap_13.to_excel('words_Chap_13.xlsx')
words_Chap_14.to_excel('words_Chap_14.xlsx') 
words_Chap_15.to_excel('words_Chap_15.xlsx') 
words_Chap_16.to_excel('words_Chap_16.xlsx') 
words_Chap_17.to_excel('words_Chap_17.xlsx') 
words_Chap_18.to_excel('words_Chap_18.xlsx') 
words_Chap_19.to_excel('words_Chap_19.xlsx') 
words_Chap_20.to_excel('words_Chap_20.xlsx') 
words_Chap_21.to_excel('words_Chap_21.xlsx') 
words_Chap_22.to_excel('words_Chap_22.xlsx') 
words_Chap_23.to_excel('words_Chap_23.xlsx') 
words_Chap_24.to_excel('words_Chap_24.xlsx') 
words_Chap_25.to_excel('words_Chap_25.xlsx') 
words_Chap_26.to_excel('words_Chap_26.xlsx') 
words_Chap_27.to_excel('words_Chap_27.xlsx') 
words_Chap_28.to_excel('words_Chap_28.xlsx') 
words_Chap_29.to_excel('words_Chap_29.xlsx') 
words_Chap_30.to_excel('words_Chap_30.xlsx') 
words_Chap_31.to_excel('words_Chap_31.xlsx') 
words_Chap_32.to_excel('words_Chap_32.xlsx') 
words_Chap_33.to_excel('words_Chap_33.xlsx') 
words_Chap_34.to_excel('words_Chap_34.xlsx') 
words_Chap_35.to_excel('words_Chap_35.xlsx') 
words_Chap_36.to_excel('words_Chap_36.xlsx') 
words_Chap_37.to_excel('words_Chap_37.xlsx') 
words_Chap_38.to_excel('words_Chap_38.xlsx') 

#Selecting top words
first_twenty_Intro = words_Intro[:20]
first_twenty_Chap_1 = words_Chap_1[:20]
first_twenty_Chap_2 = words_Chap_2[:20]
first_twenty_Chap_3 = words_Chap_3[:20]
first_twenty_Chap_4 = words_Chap_4[:20]
first_twenty_Chap_5 = words_Chap_5[:20]
first_twenty_Chap_6 = words_Chap_6[:20]
first_twenty_Chap_7 = words_Chap_7[:20]
first_twenty_Chap_8 = words_Chap_8[:20]
first_twenty_Chap_9 = words_Chap_9[:20]
first_twenty_Chap_10 = words_Chap_10[:20]
first_twenty_Chap_11 = words_Chap_11[:20]
first_twenty_Chap_12 = words_Chap_12[:20]
first_twenty_Chap_13 = words_Chap_13[:20]
first_twenty_Chap_14 = words_Chap_14[:20]
first_twenty_Chap_15 = words_Chap_15[:20]
first_twenty_Chap_16 = words_Chap_16[:20]
first_twenty_Chap_17 = words_Chap_17[:20]
first_twenty_Chap_18 = words_Chap_18[:20]
first_twenty_Chap_19 = words_Chap_19[:20]
first_twenty_Chap_20 = words_Chap_20[:20]
first_twenty_Chap_21 = words_Chap_21[:20]
first_twenty_Chap_22 = words_Chap_22[:20]
first_twenty_Chap_23 = words_Chap_23[:20]
first_twenty_Chap_24 = words_Chap_24[:20]
first_twenty_Chap_25 = words_Chap_25[:20]
first_twenty_Chap_26 = words_Chap_26[:20]
first_twenty_Chap_27 = words_Chap_27[:20]
first_twenty_Chap_28 = words_Chap_28[:20]
first_twenty_Chap_29 = words_Chap_29[:20]
first_twenty_Chap_30 = words_Chap_30[:20]
first_twenty_Chap_31 = words_Chap_31[:20]
first_twenty_Chap_32 = words_Chap_32[:20]
first_twenty_Chap_33 = words_Chap_33[:20]
first_twenty_Chap_34 = words_Chap_34[:20]
first_twenty_Chap_35 = words_Chap_35[:20]
first_twenty_Chap_36 = words_Chap_36[:20]
first_twenty_Chap_37 = words_Chap_37[:20]
first_twenty_Chap_38 = words_Chap_38[:20]

# Save the data

first_twenty_Intro.to_excel('first_twenty_Intro.xlsx')
first_twenty_Chap_1.to_excel('first_twenty_Chap_1.xlsx')
first_twenty_Chap_2.to_excel('first_twenty_Chap_2.xlsx')
first_twenty_Chap_3.to_excel('first_twenty_Chap_3.xlsx')
first_twenty_Chap_4.to_excel('first_twenty_Chap_4.xlsx')
first_twenty_Chap_5.to_excel('first_twenty_Chap_5.xlsx')
first_twenty_Chap_6.to_excel('first_twenty_Chap_6.xlsx')
first_twenty_Chap_7.to_excel('first_twenty_Chap_7.xlsx')
first_twenty_Chap_8.to_excel('first_twenty_Chap_8.xlsx')
first_twenty_Chap_9.to_excel('first_twenty_Chap_9.xlsx')
first_twenty_Chap_10.to_excel('first_twenty_Chap_10.xlsx')
first_twenty_Chap_11.to_excel('first_twenty_Chap_11.xlsx')
first_twenty_Chap_12.to_excel('first_twenty_Chap_12.xlsx')
first_twenty_Chap_13.to_excel('first_twenty_Chap_13.xlsx')
first_twenty_Chap_14.to_excel('first_twenty_Chap_14.xlsx') 
first_twenty_Chap_15.to_excel('first_twenty_Chap_15.xlsx') 
first_twenty_Chap_16.to_excel('first_twenty_Chap_16.xlsx') 
first_twenty_Chap_17.to_excel('first_twenty_Chap_17.xlsx') 
first_twenty_Chap_18.to_excel('first_twenty_Chap_18.xlsx') 
first_twenty_Chap_19.to_excel('first_twenty_Chap_19.xlsx') 
first_twenty_Chap_20.to_excel('first_twenty_Chap_20.xlsx') 
first_twenty_Chap_21.to_excel('first_twenty_Chap_21.xlsx') 
first_twenty_Chap_22.to_excel('first_twenty_Chap_22.xlsx') 
first_twenty_Chap_23.to_excel('first_twenty_Chap_23.xlsx') 
first_twenty_Chap_24.to_excel('first_twenty_Chap_24.xlsx') 
first_twenty_Chap_25.to_excel('first_twenty_Chap_25.xlsx') 
first_twenty_Chap_26.to_excel('first_twenty_Chap_26.xlsx') 
first_twenty_Chap_27.to_excel('first_twenty_Chap_27.xlsx') 
first_twenty_Chap_28.to_excel('first_twenty_Chap_28.xlsx') 
first_twenty_Chap_29.to_excel('first_twenty_Chap_29.xlsx') 
first_twenty_Chap_30.to_excel('first_twenty_Chap_30.xlsx') 
first_twenty_Chap_31.to_excel('first_twenty_Chap_31.xlsx') 
first_twenty_Chap_32.to_excel('first_twenty_Chap_32.xlsx') 
first_twenty_Chap_33.to_excel('first_twenty_Chap_33.xlsx') 
first_twenty_Chap_34.to_excel('first_twenty_Chap_34.xlsx') 
first_twenty_Chap_35.to_excel('first_twenty_Chap_35.xlsx') 
first_twenty_Chap_36.to_excel('first_twenty_Chap_36.xlsx') 
first_twenty_Chap_37.to_excel('first_twenty_Chap_37.xlsx') 
first_twenty_Chap_38.to_excel('first_twenty_Chap_38.xlsx') 

#PLOTS

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Intro[0],first_twenty_Intro[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Intro", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_1[0], first_twenty_Chap_1[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 1", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_2[0], first_twenty_Chap_2[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 2", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_3[0], first_twenty_Chap_3[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 3", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_4[0], first_twenty_Chap_4[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 4", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_5[0], first_twenty_Chap_5[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 5", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_6[0], first_twenty_Chap_6[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 6", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_7[0], first_twenty_Chap_7[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 7", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_8[0], first_twenty_Chap_8[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 8", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_9[0], first_twenty_Chap_9[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 9", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_10[0], first_twenty_Chap_10[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 10", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_11[0], first_twenty_Chap_11[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 11", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_12[0], first_twenty_Chap_12[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 12", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_13[0], first_twenty_Chap_13[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 13", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_14[0], first_twenty_Chap_14[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 14", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_15[0], first_twenty_Chap_15[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 15", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_16[0], first_twenty_Chap_16[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 16", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_17[0], first_twenty_Chap_17[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 17", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_18[0], first_twenty_Chap_18[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 18", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_19[0], first_twenty_Chap_19[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 19", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_20[0], first_twenty_Chap_20[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 20", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_21[0], first_twenty_Chap_21[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 21", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_22[0], first_twenty_Chap_22[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 22", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_23[0], first_twenty_Chap_23[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 23", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_24[0], first_twenty_Chap_24[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 24", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_25[0], first_twenty_Chap_25[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 25", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_26[0], first_twenty_Chap_26[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 26", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_27[0], first_twenty_Chap_27[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 27", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_28[0], first_twenty_Chap_28[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 28", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_29[0], first_twenty_Chap_29[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 29", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_30[0], first_twenty_Chap_30[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 30", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_31[0], first_twenty_Chap_31[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 31", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_32[0], first_twenty_Chap_32[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 32", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_33[0], first_twenty_Chap_33[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 33", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_34[0], first_twenty_Chap_34[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 34", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_35[0], first_twenty_Chap_35[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 35", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_36[0], first_twenty_Chap_36[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 36", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_37[0], first_twenty_Chap_37[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 37", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.bar(first_twenty_Chap_38[0], first_twenty_Chap_38[1], color ='#00c1d2', width = 0.4)
plt.title("Top 20 words in Chapter 38", fontsize= 24)
plt.xticks(fontsize= 22)
plt.ylabel("Count", fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

#%% ------------------ SECOND RESEARCH QUESTION (2 -	Automatically classify the report topic)

# Data preparation
Classi_data = Segn_data.drop('OPENING_CLASSIFICATION', 1)
Classi_data = Classi_data[['COMPLIANT_DESCRIPTION', 'REPORTED_TOPIC_1']]
Classi_data = Classi_data.dropna()
Classi_data['REPORTED_TOPIC_1'].value_counts()

Classi_data['REPORTED_TOPIC_1'] = Classi_data['REPORTED_TOPIC_1'].replace(['Atteggiamento operatori verso ospiti/familiari', 'Attenzione verso ospiti/familiari'], 'Accoglienza')
Classi_data['REPORTED_TOPIC_1'] = Classi_data['REPORTED_TOPIC_1'].replace(['Assistenziale', 'Infermieristico'], 'Infermieristico-assistenziale')
Classi_data['REPORTED_TOPIC_1'] = Classi_data['REPORTED_TOPIC_1'].replace(['Hotellerie-lavanderia', 'Hotellerie-igiene ambientale'], 'Lavanderia/Igiene Ambientale')
Classi_data['REPORTED_TOPIC_1'] = Classi_data['REPORTED_TOPIC_1'].replace(['Area legale','Finanza','Manutenzione', 'Servizi Comuni, servizio', 'Altro', 'Acquisti struttura', 'IT'], 'Altro')

target_classes = ['Infermieristico-assistenziale', 'Direzione-leadership', 'Accoglienza', 'Lavanderia/Igiene Ambientale', 'Area medica', 'Altro', 'Ristorazione']
Classi_data = Classi_data[Classi_data.REPORTED_TOPIC_1.isin(target_classes)]
Classi_data['REPORTED_TOPIC_1'].value_counts()

Classi_data['Class_Encode'] = Classi_data['REPORTED_TOPIC_1'].factorize()[0]


# Extracting features from text

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stop)

features = tfidf.fit_transform(Classi_data.COMPLIANT_DESCRIPTION).toarray()
labels = Classi_data.Class_Encode
features.shape

#Define sorted values
category_id_data = Classi_data[['REPORTED_TOPIC_1', 'Class_Encode']].drop_duplicates().sort_values('Class_Encode')
category_to_id = dict(category_id_data.values)
id_to_category = dict(category_id_data[['Class_Encode', 'REPORTED_TOPIC_1']].values)

# Find the terms more correlated with each topic
from sklearn.feature_selection import chi2
import numpy as np

N = 2
for REPORTED_TOPIC_1, Class_Encode in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == Class_Encode)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("### '{}':".format(REPORTED_TOPIC_1))
  print("  . Most correlated unigrams:\n. {}".format('  \n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('   \n. '.join(bigrams[-N:])))

#I check which are the models performing better

#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

train_features, test_features, train_labels, test_labels, indices_train, indices_test = train_test_split(features, labels, Classi_data.index, test_size=0.25, random_state=0)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

plt.figure(1,figsize=(24,12))
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.title("Models' accuracy", fontsize= 24)
plt.xticks(fontsize= 22)
plt.yticks(fontsize= 22)
plt.show()

# proceed with Linear SVM

model = LinearSVC()
model.fit(train_features, train_labels)
test_predicted_labels = model.predict(test_features)
conf_mat = confusion_matrix(test_labels, test_predicted_labels)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

accuracy_score(y_true = test_labels , y_pred = test_predicted_labels)

#print out the classification report
print(metrics.classification_report(test_labels, test_predicted_labels, target_names=Classi_data['REPORTED_TOPIC_1'].unique()))


#%% ----------------- THIRD RESEARCH QUESTION ( 3- 	Ranking the complaints )

Third_data = Classi_data
Third_data['Date_report'] = Segn_data['OPEN_DATE']
Third_data = Third_data.dropna()
#Remove punctuation
Third_data["COMPLIANT_DESCRIPTION"] = Third_data["COMPLIANT_DESCRIPTION"].apply(remove_punct)
#Remove stopwords
Third_data['COMPLIANT_DESCRIPTION'] = Third_data['COMPLIANT_DESCRIPTION'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))


'''
Third_data = Segn_data.drop('OPENING_CLASSIFICATION', 1)
Third_data = Third_data.dropna()
#Remove punctuation
Third_data["COMPLIANT_DESCRIPTION"] = Third_data["COMPLIANT_DESCRIPTION"].apply(remove_punct)
#Remove stopwords
Third_data['COMPLIANT_DESCRIPTION'] = Third_data['COMPLIANT_DESCRIPTION'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
'''


# Prepare one dataset per class category
Infermieristico_data = Third_data.query('REPORTED_TOPIC_1 == "Infermieristico-assistenziale"')
Infermieristico_data = Infermieristico_data.query('Date_report != "2020-02-07"') #Remove null values to avoid translation exceptions later
Direzione_data = Third_data.query('REPORTED_TOPIC_1 == "Direzione-leadership"')
Accoglienza_data = Third_data.query('REPORTED_TOPIC_1 == "Accoglienza"')
AreaMedica_data = Third_data.query('REPORTED_TOPIC_1 == "Area medica"')
Altro_data = Third_data.query('REPORTED_TOPIC_1 == "Altro"')
Ristorazione_data = Third_data.query('REPORTED_TOPIC_1 == "Ristorazione"')


#POS tagging
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in stop:
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

###  translate compliants, each dataset separately

from textblob import TextBlob

trans_text_Infermieristico = []
trans_text_Direzione = []
trans_text_Accoglienza = []
trans_text_AreaMedica = []
trans_text_Altro = []
trans_text_Ristorazione = []

###

Infermieristico_data['COMPLIANT_DESCRIPTION'][:1000]
Infermieristico_data['COMPLIANT_DESCRIPTION'][1001:2000]
Infermieristico_data['COMPLIANT_DESCRIPTION'][2001:3000]
Infermieristico_data['COMPLIANT_DESCRIPTION'][3001:]

#--------------------------------------
#Infermieristico_data = Infermieristico_data.query('Date_report != "2020-02-07"')

for i in Infermieristico_data['COMPLIANT_DESCRIPTION'][:1000]:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Infermieristico.append(trans_blob)

for i in Infermieristico_data['COMPLIANT_DESCRIPTION'][1000:2000]:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Infermieristico.append(trans_blob)

for i in Infermieristico_data['COMPLIANT_DESCRIPTION'][2000:3000]:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Infermieristico.append(trans_blob)

#Infermieristico_data = Infermieristico_data.query('Date_report != "2020-02-07"')
for i in Infermieristico_data['COMPLIANT_DESCRIPTION'][3000:]:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Infermieristico.append(trans_blob)
    
Infermieristico_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_Infermieristico

Infermieristico_data['COMPLIANT_DESCRIPTION_trans'] = Infermieristico_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

#--------------------------------------
for i in Direzione_data['COMPLIANT_DESCRIPTION']:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Direzione.append(trans_blob)
     
Direzione_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_Direzione

Direzione_data['COMPLIANT_DESCRIPTION_trans'] = Direzione_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

#--------------------------------------
for i in Accoglienza_data['COMPLIANT_DESCRIPTION']:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Accoglienza.append(trans_blob)
     
Accoglienza_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_Accoglienza

Accoglienza_data['COMPLIANT_DESCRIPTION_trans'] = Accoglienza_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

#--------------------------------------
for i in AreaMedica_data['COMPLIANT_DESCRIPTION']:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_AreaMedica.append(trans_blob)
     
AreaMedica_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_AreaMedica

AreaMedica_data['COMPLIANT_DESCRIPTION_trans'] = AreaMedica_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

#--------------------------------------
for i in Altro_data['COMPLIANT_DESCRIPTION']:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Altro.append(trans_blob)
     
Altro_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_Altro

Altro_data['COMPLIANT_DESCRIPTION_trans'] = Altro_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

#--------------------------------------
for i in Ristorazione_data['COMPLIANT_DESCRIPTION']:
    blob = TextBlob(i)
    trans_blob = blob.translate(from_lang = 'it', to = 'en')
    trans_text_Ristorazione.append(trans_blob)
     
Ristorazione_data['COMPLIANT_DESCRIPTION_trans'] = trans_text_Ristorazione

Ristorazione_data['COMPLIANT_DESCRIPTION_trans'] = Ristorazione_data['COMPLIANT_DESCRIPTION_trans'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_en)]))

### Sono arrivato qui

Infermieristico_data['POS tagged'] = Infermieristico_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)
Direzione_data['POS tagged'] = Direzione_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)
Accoglienza_data['POS tagged'] = Accoglienza_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)
AreaMedica_data['POS tagged'] = AreaMedica_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)
Altro_data['POS tagged'] = Altro_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)
Ristorazione_data['POS tagged'] = Ristorazione_data['COMPLIANT_DESCRIPTION_trans'].apply(token_pos)

###

#Third_data['POS tagged'] = Third_data['COMPLIANT_DESCRIPTION'].apply(token_pos)

#Lemmatize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

#Third_data['Lemma'] = Third_data['POS tagged'].apply(lemmatize)

Infermieristico_data['Lemma'] = Infermieristico_data['POS tagged'].apply(lemmatize)
Direzione_data['Lemma'] = Direzione_data['POS tagged'].apply(lemmatize)
Accoglienza_data['Lemma'] = Accoglienza_data['POS tagged'].apply(lemmatize)
AreaMedica_data['Lemma'] = AreaMedica_data['POS tagged'].apply(lemmatize)
Altro_data['Lemma'] = Altro_data['POS tagged'].apply(lemmatize)
Ristorazione_data['Lemma'] = Ristorazione_data['POS tagged'].apply(lemmatize)


#Polarity scores
#from textblob import TextBlob

def getPolarity(review):
        return TextBlob(review).sentiment.polarity


Infermieristico_data["Polarity_score"] = Infermieristico_data["Lemma"].apply(getPolarity)
Direzione_data["Polarity_score"] = Direzione_data["Lemma"].apply(getPolarity)
Accoglienza_data["Polarity_score"] = Accoglienza_data["Lemma"].apply(getPolarity)
AreaMedica_data["Polarity_score"] = AreaMedica_data["Lemma"].apply(getPolarity)
Altro_data["Polarity_score"] = Altro_data["Lemma"].apply(getPolarity)
Ristorazione_data["Polarity_score"] = Ristorazione_data["Lemma"].apply(getPolarity)

#Average polarities
Infermieristico_data["Polarity_score"].mean()
Direzione_data["Polarity_score"].mean()
Accoglienza_data["Polarity_score"].mean()
AreaMedica_data["Polarity_score"].mean()
Altro_data["Polarity_score"].mean()
Ristorazione_data["Polarity_score"].mean()

# PLOT TRENDS

plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Nursing/Care", fontsize=24)
plt.plot(sorted(Infermieristico_data["Date_report"]) , Infermieristico_data["Polarity_score"], linewidth=1, color='#e07204')
plt.xticks(Infermieristico_data["Date_report"][::30], Infermieristico_data["Date_report"][::30], rotation=45)
plt.axhline(y=0, color='#003c47')


plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Management ", fontsize=24)
plt.plot(sorted(Direzione_data["Date_report"]) , Direzione_data["Polarity_score"], linewidth=1, color='#00aac3')
plt.xticks(Direzione_data["Date_report"][::20], Direzione_data["Date_report"][::20], rotation=45)
plt.axhline(y=0, color='#003c47')


plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Hospitality", fontsize=24)
plt.plot(sorted(Accoglienza_data["Date_report"]) , Accoglienza_data["Polarity_score"], linewidth=1, color='#92bd1f')
plt.xticks(Accoglienza_data["Date_report"][::20], Accoglienza_data["Date_report"][::20], rotation=45)
plt.axhline(y=0, color='#003c47')


plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Medical Area", fontsize=24)
plt.plot(sorted(AreaMedica_data["Date_report"]) , AreaMedica_data["Polarity_score"], linewidth=1, color='#0c455f')
plt.xticks(AreaMedica_data["Date_report"][::20], AreaMedica_data["Date_report"][::20], rotation=45)
plt.axhline(y=0, color='#003c47')


plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Other services", fontsize=24)
plt.plot(sorted(Altro_data["Date_report"]) , Altro_data["Polarity_score"], linewidth=1, color='#f9c611')
plt.xticks(Altro_data["Date_report"][::20], Altro_data["Date_report"][::20], rotation=45)
plt.axhline(y=0, color='#003c47')


plt.figure(1,figsize=(46,12))
plt.xticks(rotation=45)
plt.title("Polarity trend for reports about Catering", fontsize=24)
plt.plot(sorted(Ristorazione_data["Date_report"]) , Ristorazione_data["Polarity_score"], linewidth=1)
plt.xticks(Ristorazione_data["Date_report"][::20], Ristorazione_data["Date_report"][::20], rotation=45)
plt.axhline(y=0, color='#003c47')
