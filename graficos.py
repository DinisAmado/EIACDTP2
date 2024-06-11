import pandas as pd
import matplotlib.pyplot as plt

# Carregar o CSV sem outliers
df_together = pd.read_csv('csv_sem_outliers.csv')  # certifique-se de que o caminho do arquivo está correto

# Gráfico do preço em função dos quartos
df_grouped_bedroom = df_together.groupby('Bedroom')['Price'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped_bedroom['Bedroom'], df_grouped_bedroom['Price'], color='skyblue')

ax.set_title('Preço Médio em Função do Número de Quartos')
ax.set_xlabel('Número de Quartos')
ax.set_ylabel('Preço Médio')

plt.show()

# Gráfico do preço em função das casas de banho
df_grouped_bathroom = df_together.groupby('Bathroom')['Price'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped_bathroom['Bathroom'], df_grouped_bathroom['Price'], color='skyblue')

ax.set_title('Preço Médio em Função do Número de Casas de Banho')
ax.set_xlabel('Número de Casas de Banho')
ax.set_ylabel('Preço Médio')

plt.show()

# Gráfico do preço em função da área
df_grouped_area = df_together.groupby('Area')['Price'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped_area['Area'], df_grouped_area['Price'], color='skyblue')

ax.set_title('Preço Médio em Função da Área')
ax.set_xlabel('Área')
ax.set_ylabel('Preço Médio')

plt.show()