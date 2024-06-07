import pandas as pd

#Ler csv 1
df1 = pd.read_csv('df1.csv')

#Ler csv2
df2 = pd.read_csv('df12.csv')

#Juntar os dois csvs e criar um novo com os dois
df_together = pd.concat([df1, df2], ignore_index=True)
df_together.to_csv('csv_combinado.csv', index = False)





