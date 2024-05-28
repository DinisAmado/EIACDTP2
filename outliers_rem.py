from df_concat import *
import numpy as np


# Função para remover outliers usando o método IQR
def remover_outliers(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[coluna] >= lower_bound) & (df[coluna] <= upper_bound)]

# Obter colunas numéricas
colunas_numericas = df_together.select_dtypes(include=[np.number]).columns

# Remover outliers das colunas numéricas
for coluna in colunas_numericas:
    df_together = remover_outliers(df_together, coluna)

# Salvar o DataFrame atualizado sem outliers em um novo arquivo CSV
#df_together.to_csv('csv_sem_outliers.csv', index=False)

# Imprimir o DataFrame atualizado
#print(df_together)
