from df_concat import *
from datetime import datetime
# Função para converter data em ano
def converter_para_ano(data):
    if isinstance(data, str):
        try:
            # Tentar analisar o formato de data completo
            return datetime.strptime(data, "%d/%m/%Y").year
        except ValueError:
            # Se falhar, assumir que é apenas o ano
            return int(float(data))  # Convertendo para float primeiro para lidar com valores como "4.4"
    elif isinstance(data, float):
        return int(data)  # Convertendo para inteiro
    elif isinstance(data, int):
        return data
    else:
        return None

# Aplicar a função à coluna 'DateSold'
df_together['DateSold'] = df_together['DateSold'].apply(converter_para_ano)

# Função para calcular as médias das colunas especificadas
def calcular_medias(df):
    media_data_de_venda = df['YearBuilt'].mean()
    media_casas_de_banho = df['Bathroom'].mean()
    media_quartos = df['Bedroom'].mean()
    return media_data_de_venda, media_casas_de_banho, media_quartos

# Calcular as médias
media_data_de_venda, media_casas_de_banho, media_quartos = calcular_medias(df_together)

# Preencher os NaNs com as médias correspondentes
df_together['YearBuilt'] = df_together['YearBuilt'].fillna(media_data_de_venda)
df_together['Bathroom'] = df_together['Bathroom'].fillna(media_casas_de_banho)
df_together['Bedroom'] = df_together['Bedroom'].fillna(media_quartos)

# Salvar o DataFrame combinado e atualizado em um novo arquivo CSV
df_together.to_csv('csv_combinado.csv', index=False)

# Imprimir o DataFrame atualizado
print(df_together)

#Ver se ainda existe missing values no csv_combinado
missing_values = df_together.isnull().sum()
#print(missing_values)


