
from transformers import pipeline

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.analyzer = pipeline("text-classification", model=self.model_name)

    def analyze_tweet(self, tweet):
        return self.analyzer(tweet)

    def interpret_result(self, result):
        label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
        return label_mapping.get(result[0]['label'], 'Unknown'), result[0]['score']