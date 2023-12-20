
from TwitterData import TwitterData
from PreprocessingTwitterData import PreprocessingTwitterData
from TwitterDataEnhancer import TwitterDataEnhancer
from TwitterDataVisualizer import TwitterDataVisualizer
from FeatureSelectorAndSplitter import FeatureSelectorAndSplitter
from TwitterSentimentAnalyzer import TwitterSentimentAnalyzer

# Ruta al archivo de datos de Twitter
file_path = 'ruta/a/tu/archivo.csv'

# Cargar y preparar los datos
twitter_data = TwitterData(file_path)
twitter_data.load_data()

preprocessor = PreprocessingTwitterData(twitter_data)
preprocessor.preprocess_data()

enhancer = TwitterDataEnhancer(preprocessor.data)
enhancer.train_word2vec()
enhancer.calculate_sentence_length()
enhancer.calculate_tweet_vectors()
enhancer.calculate_polarity()

# Visualizar los datos
visualizer = TwitterDataVisualizer(enhancer.data)
visualizer.plot_sentence_length_distribution()
visualizer.plot_sentiment_distribution()
visualizer.plot_tweet_vector_distribution(0)
visualizer.plot_feature_relationships()
visualizer.statistical_analysis()
visualizer.plot_word_cloud()

# Dividir los datos para entrenamiento y prueba
selector_splitter = FeatureSelectorAndSplitter(enhancer.data)
selector_splitter.select_features(['Entity', 'TweetContent', 'SentenceLength', 'TweetVector', 'Polarity'], 'Sentiment')
selector_splitter.split_data(test_size=0.2, random_state=42)

# Análisis de sentimientos
analyzer = TwitterSentimentAnalyzer()
tweet = "Ejemplo de tweet a analizar"
result = analyzer.analyze_tweet(tweet)
sentiment, confidence = analyzer.interpret_result(result)
print(f"Sentiment: {sentiment}, Confidence: {confidence}")