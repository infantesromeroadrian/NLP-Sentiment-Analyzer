import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class TwitterDataVisualizer:
    def __init__(self, twitter_data):
        self.data = twitter_data

    def plot_sentence_length_distribution(self):
        """
        Visualiza la distribución de la longitud de las oraciones.
        """
        fig = px.histogram(self.data, x='SentenceLength', nbins=50,
                           title='Distribución de la Longitud de las Oraciones')
        fig.update_xaxes(title='Longitud de la Oración')
        fig.update_yaxes(title='Frecuencia')
        fig.show()

    def plot_sentiment_distribution(self):
        """
        Visualiza la distribución de los sentimientos.
        """
        sentiment_cols = [col for col in self.data.columns if 'Sentiment_' in col]
        sentiment_counts = self.data[sentiment_cols].sum().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Distribución de los Sentimientos')
        fig.update_xaxes(title='Sentimiento')
        fig.update_yaxes(title='Cantidad')
        fig.show()

    def plot_tweet_vector_distribution(self, vector_index):
        """
        Visualiza la distribución de un índice específico en los vectores de tweets.
        :param vector_index: Índice en los vectores de tweets para visualizar.
        """
        vector_values = self.data['TweetVector'].apply(lambda x: x[vector_index])
        fig = px.histogram(vector_values, nbins=50, title=f'Distribución del Vector de Tweets (Índice {vector_index})')
        fig.update_xaxes(title=f'Valor de Vector (Índice {vector_index})')
        fig.update_yaxes(title='Frecuencia')
        fig.show()

    def combine_sentiments(self):
        """
        Combina las columnas de sentimiento individuales en una columna 'Sentiment'.
        """
        sentiment_cols = [col for col in self.data.columns if 'Sentiment_' in col]
        # Verifica si hay columnas de sentimiento en el DataFrame
        if not sentiment_cols:
            raise ValueError("No se encontraron columnas de sentimiento en el DataFrame.")

        # Mapeo de las columnas de sentimiento a sus etiquetas
        sentiment_mapping = {col: col.split('_')[1] for col in sentiment_cols}

        # Combinación de las columnas de sentimiento
        self.data['Sentiment'] = self.data[sentiment_cols].idxmax(axis=1)
        self.data['Sentiment'] = self.data['Sentiment'].map(sentiment_mapping)

        # Manejar filas donde todas las columnas de sentimiento están vacías
        self.data['Sentiment'] = self.data['Sentiment'].fillna('Unknown')

    def plot_feature_relationships(self):
        """
        Visualiza las relaciones entre diferentes características y sentimientos.
        """
        fig = px.scatter(self.data, x='Polarity', y='SentenceLength', color='Sentiment',
                         title='Relación entre Polaridad y Longitud de la Oración')
        fig.update_xaxes(title='Polaridad')
        fig.update_yaxes(title='Longitud de la Oración')
        fig.show()

    def statistical_analysis(self):
        """
        Realiza análisis estadísticos para identificar patrones y tendencias.
        """
        # Filtra solo las columnas numéricas para el cálculo de la correlación
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.show()

    def plot_word_cloud(self):
        """
        Genera y muestra una nube de palabras a partir del contenido de los tweets.
        """
        all_text = ' '.join(tweet for tweet in self.data['TweetContent'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de Palabras de los Tweets')
        plt.show()