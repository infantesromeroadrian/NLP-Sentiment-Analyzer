
import pandas as pd

class TwitterData:
    def __init__(self, file_path):
        """
        Constructor de la clase.
        :param file_path: Ruta al archivo de datos de Twitter.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Carga los datos desde un archivo CSV, asumiendo columnas específicas.
        """
        column_names = ['TweetID', 'Entity', 'Sentiment', 'TweetContent']
        try:
            self.data = pd.read_csv(self.file_path, header=None, names=column_names)
            print("Datos cargados exitosamente.")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")

    def preview_data(self, rows=5):
        """
        Muestra las primeras líneas del conjunto de datos.
        :param rows: Número de líneas a mostrar.
        """
        if self.data is not None:
            return self.data.head(rows)
        else:
            print("Los datos no están cargados. Por favor, use primero el método load_data.")

    def describe_data(self):
        """
        Proporciona una descripción básica del conjunto de datos.
        """
        if self.data is not None:
            return self.data.describe()
        else:
            print("Los datos no están cargados. Por favor, use primero el método load_data.")

    def info_data(self):
        """
        Muestra información general sobre el conjunto de datos, incluyendo tipos de datos y valores no nulos.
        """
        if self.data is not None:
            return self.data.info()
        else:
            print("Los datos no están cargados. Por favor, use primero el método load_data.")

    def check_nulls(self):
        """
        Verifica y cuenta los valores nulos en cada columna.
        """
        if self.data is not None:
            return self.data.isnull().sum()
        else:
            print("Los datos no están cargados. Por favor, use primero el método load_data.")