import pandas as pd

# Carregar os dados normalizados
df_normalized2 = pd.read_csv('csv_normalized2.csv')

# Calcular a correlação entre a coluna Price e as colunas Bathroom, Bedroom e YearBuilt
correlation_matrix = df_normalized2[['Price', 'Bathroom', 'Bedroom', 'Area']].corr()

print(correlation_matrix)