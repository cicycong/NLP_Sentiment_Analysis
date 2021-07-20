<h1 align="center">NLP_Sentiment_Analysis</h1>

## Description
The goal of this project is to build a sentiment classifier to classify restaurant reviews. Natural language processing techniques are applied to getting features from text.

## Getting Started
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://keras.io/)
- [tensorflow](https://www.tensorflow.org/)
- [nltk](https://www.nltk.org/)
- [gensim](https://radimrehurek.com/gensim/models/word2vec.html)

## Code
Template code is provided in the `main.py`  file. You will also be required to use the included `preprocessing.py`, `metrics.py`, and `models.py` Python files and the `reviews.csv` dataset file to complete your work. `preprocessing.py` contains modules that deal with imbalanced dataset, one hot encoding, nlp text processing, and tf-dif.  After getting the features, classifiers are trained to predict the sentiment. Three ML model are used: SGD, Random Forest, and Neural Network. The models are saved into pkl files in *output* folder and are loaded for evaluation. `metrics.py` is used for generating evaluation metrics.

## Data
The input data is stored in the *source* folder. The review data is splitted into training and validation sets and stored as training.csv and valid.csv in the *output* folder.  

**Features**
1.  `Name`: name of the clients
2. `DatePublished`: the date that the reviews are published
3. `Review`: Client's reviews

**Target Variable**
4. `RatingValue`: Client's rating of the resturant. negative (ratings 1 & 2), neutral (rating 3) and positive (ratings 4 & 5) sentiment.

