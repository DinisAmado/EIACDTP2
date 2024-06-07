import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('csv_combinado.csv')

def calcular_medias(df):
    media_area = int(round(df['Area'].mean()))
    return media_area

media_area_int = calcular_medias(df)

df['Area'] = df['Area'].replace(0, media_area_int)

df.to_csv('csv_combinado.csv', index=False)

