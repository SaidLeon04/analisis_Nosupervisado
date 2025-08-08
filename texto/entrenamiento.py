import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

# Configuración de NLTK
nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Cargar el archivo CSV
csv_file = "noticias_entrenamiento.csv"  # Cambia esto por la ruta correcta a tu archivo CSV
data = pd.read_csv(csv_file)

# Suponiendo que la columna con los títulos se llama "titulo"
titulos_noticias = data['noticia'].tolist()

# Lista de stopwords en español
spanish_stop_words = stopwords.words("spanish")

# Vectorizador TF-IDF
vectorizador = TfidfVectorizer(stop_words=spanish_stop_words)

# Transformar los títulos en vectores TF-IDF
X = vectorizador.fit_transform(titulos_noticias)

# Crear y entrenar el modelo KMeans
modelo = KMeans(n_clusters=6, random_state=1234, n_init=10)
modelo.fit(X)

# Mostrar los resultados
print(f"Cluster {modelo.labels_}")

# Imprimir los títulos y su cluster correspondiente
for i, texto in enumerate(titulos_noticias):
    print(f"{texto}: Cluster {modelo.labels_[i]}")
