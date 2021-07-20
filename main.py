
import preprocessing
import models
import metrics

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")
np.random.seed(400)

"""Load the dataset"""
df = pd.read_csv('source_data/reviews.csv', delimiter='\t')


'''check the distribution of ratings'''
#negative (ratings 1 & 2), neutral (rating 3) and positive (ratings 4 & 5) sentiment.
# print(df['RatingValue'].value_counts(ascending=True))


"""Downsample positive"""

balanced_df= preprocessing.down_sample(df, 'RatingValue', 250)
# print(balanced_df.head())


"""Dataset Split"""

# split the data into train and test set
train, validate = train_test_split(balanced_df, test_size=0.2,
                                   random_state=42, shuffle=True)
# save into csv
train.to_csv('output/data/train.csv', index=False)
validate.to_csv('output/data/validate.csv', index=False)

train = pd.read_csv('output/data/train.csv')
# train.head()

validate = pd.read_csv('output/data/validate.csv')
# validate.head()


"""Data Preprocessing"""

# transfer rating values into sentiment
train_df=preprocessing.get_sentiment(train)

'''
Stem,lemmatize and stopword
'''

# Preprocess all the headlines, storing the list of results as 'processed_docs'
train_df['Review'] = train_df['Review'].map(preprocessing.tokenize_lem)

X_train=train_df['Review'].values
y_train = train_df.drop('Review', axis=1).values


''' BOW '''
vect = CountVectorizer(token_pattern=r'\b\w+\b')
X_train = vect.fit_transform(X_train)

tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
X_train = X_train.toarray()


""" SGD model"""

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None).fit(X_train, y_train)

# save the model
joblib_file = "output/model/SGD_Model.pkl"
joblib.dump(clf, joblib_file)

# print(clf.score(X_train,y_train))

""" Random Forest"""

from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

joblib_file = "output/model/Random_Forest_Model.pkl"
joblib.dump(clf2, joblib_file)

print(clf2.score(X_train,y_train))

'''
Neural Network requires one hot encoding to y

'''

train_df=preprocessing.one_hot(train_df,'sentiment')
y_train = train_df.drop('Review', axis=1).values


nn_model=models.nn_model()
nn_model.fit(x=X_train, y=y_train, batch_size=20, epochs=10, verbose=1)

nn_model.save("output/model/nn_model.h5")

""" Validation"""

validate = pd.read_csv('output/data/validate.csv')

validate_df=preprocessing.get_sentiment(validate)


validate_df['Review'] = validate_df['Review'].map(preprocessing.tokenize_lem)

X_test=validate_df['Review'].values
y_test = validate_df.drop('Review', axis=1).values

X_test = vect.transform(X_test)
X_test = tfidf.transform(X_test)
X_test = X_test.toarray()


'''
SGD model

'''
clf = joblib.load("output/model/SGD_Model.pkl")
y_predsgd = clf.predict(X_test)

print("-----------The confusion metrix of SGD-------------")
metrics.measure_metrics(clf, X_test, y_test, y_predsgd)


'''
Random Forest

'''
clf2 = joblib.load("output/model/Random_Forest_Model.pkl")
y_predrf = clf2.predict(X_test)

print("-----------The confusion metrix of Random Forest-------------")
metrics.measure_metrics(clf2, X_test, y_test, y_predrf)


'''
Neural Network Model

'''
#Fit model on test data

validate_df=preprocessing.one_hot(validate_df,'sentiment')
y_test = validate_df.drop('Review', axis=1).values

nnclf = load_model('output/model/nn_model.h5')

y_prednn = nnclf.predict(X_test)

print("-----------The confusion metrix of Neural Network-------------")
metrics.measure_metrics(nnclf, X_test, y_test.argmax(axis=1), y_prednn.argmax(axis=1))

