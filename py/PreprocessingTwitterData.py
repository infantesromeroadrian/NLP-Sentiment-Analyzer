import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')


class PreprocessingTwitterData:
    def __init__(self, twitter_data):
        """
        Constructor de la clase.
        :param twitter_data: Instancia de la clase TwitterData con los datos cargados.
        """
        self.data = twitter_data.data
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Limpia el texto eliminando menciones, hashtags, URLs y caracteres especiales.
        :param text: Texto a limpiar.
        :return: Texto limpio.
        """
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        return text

    def normalize_text(self, text):
        """
        Normaliza el texto realizando lematización y conversión a minúsculas.
        :param text: Texto a normalizar.
        :return: Texto normalizado.
        """
        words = text.lower().split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        return ' '.join(words)

    # Creamos una funcion label_encoding para convertir las etiquetas de texto a números
    def label_encoding(self, text):
        """
        Codifica las etiquetas de texto en números.
        :param text: Texto a codificar.
        :return: Texto codificado.
        """
        if text == 'Positive':
            return 1
        elif text == 'Negative':
            return -1
        elif text == 'Neutral':
            return 0
        # Eliminamos las etiquetas que no sean positivas, negativas o neutrales
        else:
            return None

    def preprocess_data(self):
        """
        Aplica el preprocesamiento al conjunto de datos.
        """
        if self.data is not None:
            # Convertir todos los valores de TweetContent a strings
            self.data['TweetContent'] = self.data['TweetContent'].astype(str)

            # Limpieza y normalización de la columna TweetContent
            self.data['TweetContent'] = self.data['TweetContent'].apply(self.clean_text).apply(self.normalize_text)

            # Codificación de etiquetas

            self.data['Sentiment'] = self.data['Sentiment'].apply(self.label_encoding)

            # Eliminamos las filas con etiquetas no válidas

            self.data = self.data[self.data['Sentiment'].notnull()]

            # Eliminamos las filas con valores nulos en TweetContent

            self.data = self.data[self.data['TweetContent'] != '']