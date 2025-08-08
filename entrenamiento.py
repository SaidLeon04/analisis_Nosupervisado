import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans

# 1. Importar el Dataset con los datos de entrenamiento

df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")
#print(df_datos_clientes.info())
#print(df_datos_clientes.head())

# 2. Convertir el Dataframe a un Array de Numpy

X = df_datos_clientes.values
#print(X)

# 3. Enrenar el modelo
modelo = KMeans(n_clusters=2, random_state=1234,n_init=10)
modelo.fit(X)

# 4. Analisis del modelo
df_datos_clientes['cluster'] = modelo.labels_
analisis = df_datos_clientes.groupby('cluster').mean()
print(analisis)


# 5. Exportar el modelo
joblib.dump(modelo, "modelo_segmentacion_clientes.pkt")

# 6. Graficar los clientes
centroides = modelo.clusters_centers_
etiquetas = modelo.labels_

cluster0 = X[etiquetas == 0]
cluster1= X[etiquetas == 1]
cluster2= X[etiquetas == 2]

# colocar los puntos de cada cluster 
plt.scatter(cluster0[:,0]cluster0[:,1], c='red', label='Temporada')

plt.scatter(cluster1[:,0]cluster1[:,1], c='blue', label='VIP')

plt.scatter(cluster2[:,0]cluster2[:,1], c='green', label='Ofertas')

# colocar los centroides de cada cluster
plt.scatter(centroides[:,0]centroides[:,1], c='black', marker='x' label='Centroides')

# colocar titulos y etiquetas
plt.title('Seguimiento de clientes')
plt.xlabel('Gasto total')
plt.ylabel('Visitas')
plt.legend()
plt.grid(True)

plt.savefig('graficas/clusters.png')