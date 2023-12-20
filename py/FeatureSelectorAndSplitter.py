
from sklearn.model_selection import train_test_split

class FeatureSelectorAndSplitter:
    def __init__(self, data):
        """
        Constructor de la clase.
        :param data: DataFrame con las características y etiquetas.
        """
        self.data = data

    def select_features(self, feature_columns, target_column):
        """
        Selecciona las columnas de características y la columna objetivo.
        :param feature_columns: Lista de columnas a usar como características.
        :param target_column: Columna objetivo.
        """
        self.features = self.data[feature_columns]
        self.target = self.data[target_column]

    def split_data(self, test_size=0.2, random_state=None):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        :param test_size: Proporción del conjunto de prueba.
        :param random_state: Semilla para la generación de números aleatorios.
        :return: Conjuntos de datos de entrenamiento y prueba.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )