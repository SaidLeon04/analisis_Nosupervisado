import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

csv_file = "noticias_entrenamiento.csv" 
data = pd.read_csv(csv_file)

titulos_noticias = data['noticia'].tolist()

spanish_stop_words = stopwords.words("spanish")

vectorizador = TfidfVectorizer(stop_words=spanish_stop_words)

X = vectorizador.fit_transform(titulos_noticias)

modelo = KMeans(n_clusters=10, random_state=1234, n_init=3000)
modelo.fit(X)

print(f"Cluster {modelo.labels_}")

for i, texto in enumerate(titulos_noticias):
    print(f"{texto}: Cluster {modelo.labels_[i]}")
