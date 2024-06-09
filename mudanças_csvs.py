import pandas as pd

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv('melb_data.csv')

# Remover colunas do DataFrame
colunas_para_remover = ['Suburb','Address','Type','Method','SellerG','Date','Distance','Postcode','Bedroom2','Car','BuildingArea','YearBuilt','CouncilArea','Lattitude','Longtitude','Regionname','Propertycount']  # Lista das colunas a serem removidas
df = df.drop(colunas_para_remover, axis=1)

# Dicionário para renomear as colunas
novos_nomes = {
    'Rooms': 'Bedroom',
    'Landsize': 'Area'
}

# Renomear as colunas
df = df.rename(columns=novos_nomes)

# Lista com a nova ordem das colunas
nova_ordem = ['Price', 'Area', 'Bedroom', 'Bathroom']
# Ajuste conforme as colunas existentes no seu DataFrame

# Reordenar as colunas
df = df[nova_ordem]

# Salvar o DataFrame resultante de volta para um arquivo CSV, se necessário
df.to_csv('df1.csv', index=False)

df2 = pd.read_csv('Housing.csv')

colunas_para_remover2 = ['id','date','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15']  # Lista das colunas a serem removidas
df2 = df2.drop(colunas_para_remover2, axis=1)

novos_nomes = {
    'price': 'Price',
    'bedrooms': 'Bedroom',
    'bathrooms': 'Bathroom',
    'sqft_lot15': 'Area'
}

# Renomear as colunas
#df2 = df2.rename(columns=novos_nomes)

# Renomear as colunas
df2 = df2.rename(columns=novos_nomes)

# Lista com a nova ordem das colunas
nova_ordem = ['Price', 'Area', 'Bedroom', 'Bathroom']
# Ajuste conforme as colunas existentes no seu DataFrame

# Reordenar as colunas
df2 = df2[nova_ordem]

# Salvar o DataFrame resultante de volta para um arquivo CSV, se necessário
df2.to_csv('df12.csv', index=False)




#missing_values = df2.isnull().sum()
#print(missing_values)


