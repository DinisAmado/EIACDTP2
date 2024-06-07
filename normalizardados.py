import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carregar o CSV sem outliers
df = pd.read_csv('csv_sem_outliers.csv')

# Colunas a serem normalizadas
colunas_para_normalizar = ["Price", "Bedroom", "Bathroom", "Area"]

# Inicializar o MinMaxScaler
scaler = MinMaxScaler()

# Normalizar as colunas
df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

# Salvar o dataframe normalizado em um novo CSV
df.to_csv('csv_normalized2.csv', index=False)

print("Dados normalizados e salvos em 'csv_normalized2.csv'")
