import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carregar o CSV sem outliers
df = pd.read_csv('csv_sem_outliers.csv')

# Definir as features (X) e a variável alvo (y)
features = ['Bedroom', 'Bathroom', 'DateSold', 'YearBuilt', 'Price']
X = df[features]

# Padronizar as features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Converter de volta para um DataFrame
df_normalized = pd.DataFrame(X_normalized, columns=features)

# Salvar o DataFrame normalizado em um novo arquivo CSV
df_normalized.to_csv('csv_normalized.csv', index=False)

# Verificar os primeiros registros do DataFrame normalizado
print(df_normalized.head())
