import pandas as pd
import numpy as np
from scipy import stats

# Carregar os dados do arquivo CSV
df = pd.read_csv('csv_combinado.csv')

# Passo 1: Calcular o Z-score para as colunas 'Price' e 'Area'
df['Z_score_Price'] = np.abs(stats.zscore(df['Price']))
df['Z_score_Area'] = np.abs(stats.zscore(df['Area']))

# Passo 2: Definir um limite para o Z-score
threshold = 3

# Passo 3: Filtrar os dados para remover outliers das colunas 'Price' e 'Area'
df_no_outliers = df[(df['Z_score_Price'] < threshold) & (df['Z_score_Area'] < threshold)]

# Remover as colunas 'Z_score' que não são mais necessárias
df_no_outliers = df_no_outliers.drop(columns=['Z_score_Price', 'Z_score_Area'])

# Aplicar a função de remoção de outliers para cada coluna ('Bathroom' e 'Bedroom') usando IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    return df_filtered

# Aplicar a função para remover outliers para as colunas 'Bathroom' e 'Bedroom'
columns_to_filter = ['Bathroom', 'Bedroom']
for column in columns_to_filter:
    df_no_outliers = remove_outliers_iqr(df_no_outliers, column)

# Salvar o DataFrame atualizado sem outliers em um novo arquivo CSV
df_no_outliers.to_csv('csv_sem_outliers.csv', index=False)

# Imprimir o DataFrame atualizado
print(df_no_outliers)
