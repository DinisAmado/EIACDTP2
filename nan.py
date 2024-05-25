from df_concat import *
import numpy as np

def calcular_medias(df):
    media_data_de_venda = df['YearBuilt'].mean()
    media_casas_de_banho = df['Bathroom'].mean()
    media_quartos = df['Bedroom'].mean()
    return media_data_de_venda, media_casas_de_banho, media_quartos

media_data_de_venda, media_casas_de_banho, media_quartos = calcular_medias(df_together)

# Preencher os NaNs com as médias correspondentes
df_together['YearBuilt'].fillna(media_data_de_venda, inplace=True)
df_together['Bathroom'].fillna(media_casas_de_banho, inplace=True)
df_together['Bedroom'].fillna(media_quartos, inplace=True)

# Salvar o DataFrame combinado e atualizado em um novo arquivo CSV
df_together.to_csv('csv_combinado.csv', index=False)

# Imprimir o DataFrame atualizado
print(df_together)