# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:08:31 2020

@author: win
"""

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

## Loading the large dataset consisting of 24 rows ##
md = pd. read_csv('../input/movies_metadata.csv')
md.head()

## Extracting Genre Values from Genre Column as genre now is [{'id': 16, 'name': 'Animation'}, {'id': 35, '..##
## After running the code Genre becomes Animation, Comedy , ##

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

## Reading Small data set 
links_small = pd.read_csv('../input/links_small.csv')

## Extracing only the not null tmbid and cnverting it to integer ##
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
links_small.head()

## Deleting specific rows
md = md.drop([19730, 29503, 35587])

# converting the id of md dataframe to integer ##
md['id'] = md['id'].astype('int')

## now selecting only those rows from md whose Id is present in links_Small dataframe
smd = md[md['id'].isin(links_small)]
smd.shape

## reset_index() method sets a list of integer ranging from 0 to length of data as index ##
smd = smd.reset_index()

## extract titles ##
titles = smd['title']

## Combining tagline and overview to description ##

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

## TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

## to get the tf idf value for each word we can use this but in pur case will consume a lot of memory 
feature_names = tf.get_feature_names()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.head()

## Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's linear_kernel instead of cosine_similarities since it is much faster.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)   ## We now have a pairwise cosine similarity matrix for all the movies in our dataset. 

## Collaborative Filtering
reader = Reader()
ratings = pd.read_csv('../input/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
trainset = data.build_full_trainset()
svd.train(trainset)


## will give a series with index as title and data as index of smd i.e. 0,1,2,3,...,9098 ## So when we pass indices['Avatar'] we get value 7973 i.e. the row no. basically
indices = pd.Series(smd.index, index=smd['title']) ## pandas.Series( data, index, dtype, copy) ##

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
## now reaading from small dataset 
id_map = pd.read_csv('../input/links_small.csv')[['movieId', 'tmdbId']]

# converting the tmbid column to int ##
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)

# renaming the cols of dataframe id_map ##
id_map.columns = ['movieId', 'id']

## Merging the smd's title column to id_map df and index is title 
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')

## Returns id as indices and movie ids col 
indices_map = id_map.set_index('id')


def hybrid(userId, title):
    ## Find the index of the movie title i.e. row no.
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    ## Pass the index value ..find the cosine similarity for that index value and then enumerate to find index and value of similarity arrange them in list
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    ## The lambda function takes input x return x[1] which is the second element of x.sort(mylist, key=lambda x: x[1]) sorts mylist based on the value of key as applied to each element of the list.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    ## return top 25 rows
    sim_scores = sim_scores[1:26]
    ## returns only indices value
    movie_indices = [i[0] for i in sim_scores]
    ## returns 'title', 'vote_count', 'vote_average', 'year', 'id' from smd having similar movie indices
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
