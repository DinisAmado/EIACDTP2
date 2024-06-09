import pandas as pd

# Carregar os dados normalizados
df_normalized = pd.read_csv('normalized.csv')

# Calcular a correlação entre a coluna Price e as colunas Bathroom, Bedroom e YearBuilt
correlation_matrix = df_normalized[['Price', 'Bathroom', 'Bedroom', 'Area']].corr()

print(correlation_matrix)