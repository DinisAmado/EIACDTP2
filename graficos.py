import pandas as pd
import matplotlib.pyplot as plt

#Talvez este ficheiro não fique até ao fim do trabalho dependendo do pretendido

# Carregar o CSV sem outliers
df_together = pd.read_csv('csv_sem_outliers.csv')


#Gráfico do preço em função dos quartos:

# Agrupar os dados pelo número de quartos e calcular a média dos preços
df_grouped = df_together.groupby('Bedroom')['Price'].mean().reset_index()

# Criar o gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped['Bedroom'], df_grouped['Price'], color='skyblue')

# Adicionar título e rótulos
ax.set_title('Preço Médio em Função do Número de Quartos')
ax.set_xlabel('Número de Quartos')
ax.set_ylabel('Preço Médio')

# Mostrar o gráfico
plt.show()


#Gráfico para ver o preço em função das casas de banho:

df_grouped = df_together.groupby('Bathroom')['Price'].mean().reset_index()

# Criar o gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped['Bathroom'], df_grouped['Price'], color='skyblue')

# Adicionar título e rótulos
ax.set_title('Preço Médio em Função do Número de Casas de Banho')
ax.set_xlabel('Número de Casas De Banho')
ax.set_ylabel('Preço Médio')

# Mostrar o gráfico
plt.show()


#Gráfico para ver o preço em função dos anos de construção:

df_grouped = df_together.groupby('Area')['Price'].mean().reset_index()

# Criar o gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped['Area'], df_grouped['Price'], color='skyblue')
