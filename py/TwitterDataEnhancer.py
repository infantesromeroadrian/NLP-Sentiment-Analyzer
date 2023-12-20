import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk

nltk.download('punkt')

class TwitterDataEnhancer:
    def __init__(self, twitter_data):
        """
        Constructor de la clase.
        :param twitter_data: DataFrame con la columna 'TweetContent'.
        """
        self.data = twitter_data
        self.model_w2v = None

    def train_word2vec(self):
        """
        Entrena un modelo Word2Vec con los tweets.
        """
        tweets = self.data['TweetContent'].apply(word_tokenize).tolist()
        self.model_w2v = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)

    def calculate_sentence_length(self):
        """
        Calcula la longitud de las oraciones en 'TweetContent'.
        """
        self.data['SentenceLength'] = self.data['TweetContent'].apply(lambda x: len(word_tokenize(x)))

    def _tweet_vector(self, tweet):
        """
        Calcula el vector promedio de un tweet.
        :param tweet: Tweet individual.
        :return: Vector promedio del tweet.
        """
        vector = [self.model_w2v.wv[word] for word in word_tokenize(tweet) if word in self.model_w2v.wv]
        if len(vector) == 0:
            return np.zeros(100)
        return np.mean(vector, axis=0)

    def calculate_tweet_vectors(self):
        """
        Calcula y almacena los vectores promedio de todos los tweets en 'TweetContent'.
        """
        self.data['TweetVector'] = self.data['TweetContent'].apply(self._tweet_vector)

    def calculate_cosine_similarity(self, index1, index2):
        """
        Calcula la similitud de coseno entre dos tweets.
        :param index1: Índice del primer tweet.
        :param index2: Índice del segundo tweet.
        :return: Similitud de coseno.
        """
        vector_a = self.data['TweetVector'].iloc[index1].reshape(1, -1)
        vector_b = self.data['TweetVector'].iloc[index2].reshape(1, -1)
        return cosine_similarity(vector_a, vector_b)[0][0]

    def calculate_polarity(self):
        """
        Calcula la polaridad de los tweets en 'TweetContent'.
        """
        self.data['Polarity'] = self.data['TweetContent'].apply(lambda x: TextBlob(x).sentiment.polarity)