
import pandas as pd
# !pip install -U gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk

nltk.download('wordnet')
stemmer = SnowballStemmer("english")


def down_sample(df, target, nums):
  '''

  Down sample the imbalanced dataset

  '''

  neg_neutral = df[df[target].isin([1,2,3])]
  pos_data = df[df[target].isin([4,5])]

  #Randomly select n observations from the postive
  pos_data_drop = pos_data.sample(n=nums,random_state=42)
  pos_data_drop[target].value_counts(ascending=True)

  #concatenate data
  balanced_df = pd.concat([neg_neutral, pos_data_drop])
  balanced_df[target].value_counts(ascending=True)
  return balanced_df



def get_sentiment(df):
  '''

  transfer rating values into sentiment

  '''

  rate_guide = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
  df['sentiment'] = df['RatingValue'].map(rate_guide)

  # keep only sentiment and review column
  df = df[['Review', 'sentiment']]
  return df



def one_hot(df, column):
  '''

  One hot encoding to the categorical variables.

  '''
  one_hot = pd.get_dummies(df[column])
  df.drop([column], axis=1, inplace=True)
  onehot_df = pd.concat([df, one_hot], axis=1)
  return onehot_df


'''
Write a function to perform the pre-processing steps on the entire dataset
'''

def lemmatize_stemming(text):
  return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def tokenize_lem(text):
  result = ''
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
      result += ' ' + lemmatize_stemming(token)

  return result


