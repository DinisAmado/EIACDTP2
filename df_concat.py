import pandas as pd

#Ler csv 1
df_all_perth = pd.read_csv('all_perth.csv')

#Ler csv2
df_property_sales = pd.read_csv('Property_Sales.csv')

#Juntar os dois csvs e criar um novo com os dois
df_together = pd.concat([df_all_perth, df_property_sales], ignore_index=True)
df_together.to_csv('csv_combinado.csv', index = False)





