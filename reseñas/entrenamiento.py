import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer # libreria para reducir palabras a su raíz (buenisimo === bueno)

stemmer = SpanishStemmer() # inicizaliación del lematizador

nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

csv_file = "reseñas_entrenamiento.csv" 
data = pd.read_csv(csv_file, sep="|")

reseñas = data['reseña'].tolist()
# reseñas = [" ".join(stemmer.stem(p) for p in r.split()) for r in reseñas]


spanish_stop_words = stopwords.words("spanish")

vectorizador = TfidfVectorizer(stop_words=spanish_stop_words)

X = vectorizador.fit_transform(reseñas)

modelo = KMeans(n_clusters=2, random_state=1234, n_init=3000)
modelo.fit(X)

print(f"Cluster {modelo.labels_}")

for i, texto in enumerate(reseñas):
    print(f"{texto}: Cluster {modelo.labels_[i]}")
