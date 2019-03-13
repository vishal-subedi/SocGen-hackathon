from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import re
from datetime import datetime
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import xgboost
from lexical_diversity import lex_div as ld
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score as fb, f1_score as f1, silhouette_score as shs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from sklearn.cluster import KMeans, hierarchical 
from sklearn.metrics.pairwise import euclidean_distances
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from textblob import TextBlob, Word
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split as tts
stop_words = stopwords.words('english')
ss = StandardScaler()
pca = PCA(5)
le = LabelEncoder()
imp = Imputer(strategy = 'most_frequent')
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('Desktop/Datasets/Brainwaves/news.csv', header = 0, index_col = None)

for i in range(df.shape[0]):
    df.iloc[i, 1] = df.iloc[i, 1].lower()
    
for i in range(df.shape[0]):
    df.iloc[i, 2] = df.iloc[i, 2].lower()
    
    
def remove_punc(sent):
    table = str.maketrans({key: None for key in string.punctuation})
    return sent.translate(table)           



for i in range(df.shape[0]):
    df.iloc[i, 1] = remove_punc(df.iloc[i, 1])
    df.iloc[i, 2] = remove_punc(df.iloc[i, 2])
    
    
    
def clean_sent(sentence): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex stat ements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence).split())
    
for i in range(df.shape[0]):
    df.iloc[i, 1] = clean_sent(df.iloc[i, 1])
    df.iloc[i, 2] = clean_sent(df.iloc[i, 2])
    
    
    
    
def tokenize_and_stem(sent):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    filtered = [w for w in words if w not in stop_words]
    stems = [stemmer.stem(t) for t in filtered]
    return stems


head_final = df.iloc[:, 1].apply(lambda x : ' '.join([word for word in x.split() if word not in stop_words]))
text_final = df.iloc[:, 2].apply(lambda x : ' '.join([word for word in x.split() if word not in stop_words]))

X = pd.DataFrame({"head": head_final, "text": text_final})


tfidf = TfidfVectorizer(max_features = 200, use_idf = True, stop_words = 'english', tokenizer = tokenize_and_stem)

tfidf_matrix = tfidf.fit_transform(X.iloc[:, 1])

terms = []

terms = tfidf.get_feature_names()

km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 1, verbose = 0, random_state = 3425)
km.fit(tfidf_matrix)
labels = km.labels_
clusters = labels.tolist()

score = shs(tfidf_matrix, labels)

datum = pd.DataFrame([tfidf_matrix.toarray()[i] for i in range(3000)])
clusters = pd.DataFrame([km.cluster_centers_[i] for i in range(5)])

dist = euclidean_distances(datum, clusters)                
    
pd.DataFrame({"id": df.iloc[:, 0], "cluster": labels}).to_csv('brainwaves_2.csv', index = None, columns = ["id", "cluster"])

np.savetxt('brainwaves.txt', dist)