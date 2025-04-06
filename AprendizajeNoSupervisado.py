import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_transporte_masivo.csv")

le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_transporte = LabelEncoder()

df['origen'] = le_origen.fit_transform(df['origen'])
df['destino'] = le_destino.fit_transform(df['destino'])
df['transporte'] = le_transporte.fit_transform(df['transporte'])

X = df[['origen', 'destino', 'distancia_km', 'transporte', 'hora_pico', 'tiempo_total_min', 'precio_total_cop']]

modelo = IsolationForest(contamination=0.05, random_state=42)
df['anomalía'] = modelo.fit_predict(X)

rutas_atipicas = df[df['anomalía'] == -1].copy()

rutas_atipicas['origen'] = le_origen.inverse_transform(rutas_atipicas['origen'])
rutas_atipicas['destino'] = le_destino.inverse_transform(rutas_atipicas['destino'])
rutas_atipicas['transporte'] = le_transporte.inverse_transform(rutas_atipicas['transporte'])

print("Rutas atípicas detectadas:")
print(rutas_atipicas[['origen', 'destino', 'distancia_km', 'tiempo_total_min', 'precio_total_cop']])

plt.scatter(df['distancia_km'], df['precio_total_cop'], c=df['anomalía'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Distancia (km)")
plt.ylabel("Precio (COP)")
plt.title("Detección de rutas atípicas")
plt.show()