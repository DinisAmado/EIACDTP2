import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('csv_combinado.csv')

# Função para remover outliers baseados no Z-Score
def rem_outliers_zscore(df, column_name, n_std_devs):
    #Encontrar os limites 
    upper_limit = df[column_name].mean() + n_std_devs * df[column_name].std()
    lower_limit = df[column_name].mean() - n_std_devs * df[column_name].std()

    #Trimming => removing dados outliers
    new_df = df.loc[(df[column_name] < upper_limit) & (df[column_name] > lower_limit)]

    return upper_limit, lower_limit, new_df

def rem_outliers_price(df):
    #plt.hist(df['Price'], bins=200, color='skyblue', edgecolor='black')
    #plt.show()


    #Encontrar os limites 
    print(df['Price'].std())

    upper_limit, lower_limit, new_df = rem_outliers_zscore(df, 'Price', 1.5)
    print("Upper limit: ", upper_limit)
    print("Lower limit: ", lower_limit)

    print('before removing outliers:', len(df))
    print('after removing outliers:', len(new_df))
    print('outliers:', len(df) - len(new_df))
    #plt.hist(new_df['Price'], bins=100, color='skyblue', edgecolor='black')
    #plt.show()

    upper_limit, lower_limit, new_df = rem_outliers_zscore(new_df, 'Price', 2.0)
    
    plt.hist(new_df['Price'], bins=100, color='skyblue', edgecolor='black')
    plt.show()
    return new_df

def rem_outliers_bedroom(df):

    #plt.hist(df['Bedroom'],bins=50, color='skyblue', edgecolor='black')
    #plt.show()
    for i in range(1,21,1):
        print("[{0}%-{1}%]:{2}".format((i-1)*5, i*5,df['Bedroom'].quantile(float(i)/20.0)))

    upper_limit = 5
    new_df = df.loc[(df['Bedroom'] <= upper_limit)]

    plt.hist(new_df['Bedroom'],bins=10, color='skyblue', edgecolor='black')
    plt.show()
 
    return new_df

def rem_outliers_bath(df):
    #plt.hist(df['Bathroom'],bins=50, color='skyblue', edgecolor='black')
    #plt.show()

    for i in range(1,21,1):
        print("[{0}%-{1}%]:{2}".format((i-1)*5, i*5,df['Bathroom'].quantile(float(i)/20.0)))

    upper_limit = 3
    lower_limit = 1
    new_df = df.loc[(df['Bathroom'] <= upper_limit) & (df['Bathroom'] >= lower_limit)]

    plt.hist(new_df['Bathroom'],bins=10, color='skyblue', edgecolor='black')
    plt.show()

    return new_df

def rem_outliers_area(df):
    #plt.hist(df['Area'], bins=100, color='skyblue', edgecolor='black')
    #plt.show()

    #Encontrar os limites 
    print(df['Area'].std())

    upper_limit, lower_limit, new_df = rem_outliers_zscore(df, 'Area', 0.5)
    print("Upper limit: ", upper_limit)
    print("Lower limit: ", lower_limit)

    
    print('before removing outliers:', len(df))
    print('after removing outliers:', len(new_df))
    print('outliers:', len(df) - len(new_df))
    #plt.hist(new_df['Area'], bins=50, color='skyblue', edgecolor='black')
    #plt.show()

    upper_limit, lower_limit, new_df = rem_outliers_zscore(new_df, 'Area', 1)
    plt.hist(new_df['Area'], bins=50, color='skyblue', edgecolor='black')
    plt.show()

    return new_df

# Chamada das funções

print("Dataset size: {0}".format(len(df)))
df_remout = rem_outliers_price(df)
print("Dataset size: {0}".format(len(df_remout)))
df_remout = rem_outliers_bedroom(df_remout)
print("Dataset size: {0}".format(len(df_remout)))
df_remout = rem_outliers_bath(df_remout)
print("Dataset size: {0}".format(len(df_remout)))
df_remout = rem_outliers_area(df_remout)
print("Dataset size: {0}".format(len(df_remout)))

#Normalizar os dados

df_remout.to_csv('csv_sem_outliers.csv', index=False)
clear_df = pd.read_csv('csv_sem_outliers.csv', low_memory=False)


#Normalização dos dados
min_max_scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(clear_df), columns=clear_df.columns)

df_min_max_scaled.to_csv('normalized.csv', index=False)
final_df = pd.read_csv('normalized.csv', low_memory=False)

# print('Dados sem outliers e normalizados: ')
print(final_df)