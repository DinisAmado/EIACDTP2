import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv('csv_sem_outliers.csv')

# Plotar boxplot para cada coluna numérica
for column in df.columns:
    if df[column].dtype != 'object':  # Verificar se é uma coluna numérica
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot para {column}')
        plt.show()
