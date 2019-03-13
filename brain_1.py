from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()
import pandas as pd
import re
from datetime import datetime
import xgboost
from lexical_diversity import lex_div as ld
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score as fb, f1_score as f1
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from textblob import TextBlob, Word
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split as tts
stop = stopwords.words('english')
ss = StandardScaler()
pca = PCA(5)
le = LabelEncoder()
imp = Imputer(strategy = 'most_frequent')


df_train = pd.read_csv('Desktop/Datasets/Brainwaves/train.csv', header = 0, index_col = None)

df_test = pd.read_csv('Desktop/Datasets/Brainwaves/test.csv', header = 0, index_col = None)

df_train_pilot = df_train.sample(frac = 1).reset_index(drop = True)

df_train_pilot.loc[:, 'Transaction-Type'] = le.fit_transform(df_train_pilot.loc[:, 'Transaction-Type'])

df_train_pilot.loc[:, 'Complaint-reason'] = le.fit_transform(df_train_pilot.loc[:, 'Complaint-reason'])

df_train_pilot.loc[:, 'Complaint-Status'] = le.fit_transform(df_train_pilot.loc[:, 'Complaint-Status'])




nans = []
for i in range(df_train_pilot.shape[0]):
    if df_train_pilot.loc[i, 'Consumer-disputes'] == 'Yes':
        df_train_pilot.loc[i, 'Consumer-disputes'] = 1
    elif df_train_pilot.loc[i, 'Consumer-disputes'] == 'No':
        df_train_pilot.loc[i, 'Consumer-disputes'] = 0
    else:
        nans.append(i)
        
for i in nans:
    df_train_pilot.loc[i, 'Consumer-disputes'] = df_train_pilot.loc[:, 'Consumer-disputes'].mode()[0]

def days(d1, d2):
    date_format = '%m/%d/%Y'
    return((abs(datetime.strptime(d1, date_format) - datetime.strptime(d2, date_format))).days)
    
for i in range(df_train_pilot.shape[0]):
    df_train_pilot.loc[i, 'days'] = days(df_train_pilot.loc[i, 'Date-received'], df_train_pilot.loc[i, 'Date-sent-to-company'])
    
    
    
def clean_sent(sentence): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence).split())    
    

for i in range(df_train_pilot.shape[0]):
    df_train_pilot.loc[i, "Consumer-complaint-summary"] = clean_sent(df_train_pilot.loc[i, "Consumer-complaint-summary"])
    

sent = []
for i in range(df_train_pilot.shape[0]):
    sent.append(Word(df_train_pilot.loc[i, "Consumer-complaint-summary"]).lemmatize())
    

polarity = []
for i in range(len(sent)):
    polarity.append(TextBlob(sent[i]).sentiment[0])
    
subjectivity = []
for i in range(len(sent)):
    subjectivity.append(TextBlob(sent[i]).sentiment[1])
    

def avg_word(sentence):
    words = sentence.split()
    if len(words) != 0:
        return (sum(len(word) for word in words)/len(words))
    else:
        return 0

avg = []
for i in range(len(sent)):
    avg.append(avg_word(sent[i]))
    
no_stop = []
count = 0
for i in sent:
    for j in i.split():
        if j in stop:
            count = count + 1
    no_stop.append(count)
    count = 0
    
lex = []
for i in range(len(sent)):
    lex.append(ld.ttr(sent[i]))

ser_1= pd.Series(subjectivity)
ser_2 = pd.Series(polarity)
ser_3 = pd.Series(avg)
ser_4 = pd.Series(no_stop)
ser_5 = pd.Series(lex)
df_train_1 = pd.concat([df_train_pilot, ser_1, ser_2, ser_3, ser_4, ser_5], axis = 1)

    
X = df_train_1.iloc[:, [2, 3, 7, 9, 10, 11, 12, 13, 14]]
y = df_train_1.iloc[:, 6]

X_train, X_test, y_train, y_test = tts(X, y, test_size = .33, random_state = 1)

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

lr = LogisticRegression(random_state = 1)
lr.fit(X_train, y_train)

f1(y_test, lr.predict(X_test), average = 'micro')

knc = KNeighborsClassifier(n_neighbors = 10, p = 3)
knc.fit(X_train, y_train)

f1(y_test, knc.predict(X_test), average = 'micro')

rfc = RandomForestClassifier(n_estimators = 50, criterion = 'gini', random_state = 1, max_features = 'log2')
rfc.fit(X_train, y_train)

f1(y_test, rfc.predict(X_test), average = 'micro')

gb = GradientBoostingClassifier(random_state = 1)
gb.fit(X_train, y_train)

f1(y_test, gb.predict(X_test), average = 'micro')


nb = GaussianNB()
nb.fit(X_train, y_train)

f1(y_test, nb.predict(X_test), average = 'micro')


xgb = xgboost.XGBClassifier(random_state = 0)
xgb.fit(X_train, y_train)

f1(y_test, xgb.predict(X_test), average = 'micro')





df_test.loc[:, 'Transaction-Type'] = le.fit_transform(df_test.loc[:, 'Transaction-Type'])

df_test.loc[:, 'Complaint-reason'] = le.fit_transform(df_test.loc[:, 'Complaint-reason'])




nans_test = []
for i in range(df_test.shape[0]):
    if df_test.loc[i, 'Consumer-disputes'] == 'Yes':
        df_test.loc[i, 'Consumer-disputes'] = 1
    elif df_test.loc[i, 'Consumer-disputes'] == 'No':
        df_test.loc[i, 'Consumer-disputes'] = 0
    else:
        nans_test.append(i)
        
for i in nans_test:
    df_test.loc[i, 'Consumer-disputes'] = df_test.loc[:, 'Consumer-disputes'].mode()[0]
    
for i in range(df_test.shape[0]):
    df_test.loc[i, 'days'] = days(df_test.loc[i, 'Date-received'], df_test.loc[i, 'Date-sent-to-company'])
        


for i in range(df_test.shape[0]):
    df_test.loc[i, "Consumer-complaint-summary"] = clean_sent(df_test.loc[i, "Consumer-complaint-summary"])
    

sent_test = []
for i in range(df_test.shape[0]):
    sent_test.append(Word(df_test.loc[i, "Consumer-complaint-summary"]).lemmatize())
    

polarity_test = []
for i in range(len(sent_test)):
    polarity_test.append(TextBlob(sent_test[i]).sentiment[0])
    
subjectivity_test = []
for i in range(len(sent_test)):
    subjectivity_test.append(TextBlob(sent_test[i]).sentiment[1])
    

avg_test = []
for i in range(len(sent_test)):
    avg_test.append(avg_word(sent_test[i]))
    
no_stop_test = []
count = 0
for i in sent_test:
    for j in i.split():
        if j in stop:
            count = count + 1
    no_stop_test.append(count)
    count = 0
    
lex_test = []
for i in range(len(sent_test)):
    lex_test.append(ld.ttr(sent_test[i]))

ser_1= pd.Series(subjectivity_test)
ser_2 = pd.Series(polarity_test)
ser_3 = pd.Series(avg_test)
ser_4 = pd.Series(no_stop_test)
ser_5 = pd.Series(lex_test)
df_test_1 = pd.concat([df_test, ser_1, ser_2, ser_3, ser_4, ser_5], axis = 1)


X_Test = df_test_1.iloc[:, [2, 3, 6, 8, 9, 10, 11, 12, 13]]


X_Test = ss.fit_transform(X_Test)


X_Test = pca.fit_transform(X_Test)

pred = gb.predict(X_Test)

pred_string = []

for i in range(len(pred)):
    if pred[i] == 0:
        pred_string.append('Closed')
    elif pred[i] == 1:
        pred_string.append('Closed with explanation')
    elif pred[i] == 2:
        pred_string.append('Closed with monetary relief')
    elif pred[i] == 3:
        pred_string.append('Closed with non-monetary relief')
    elif pred[i] == 4:
        pred_string.append('Untimely response')
        
        
    
pd.DataFrame({"Complaint-ID": df_test_1.loc[:, 'Complaint-ID'], "Complaint-Status": pred_string}).to_csv('brainwaves.csv', index = None)