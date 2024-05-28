import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Carregar o CSV sem outliers
df = pd.read_csv('csv_sem_outliers.csv')

# Definir as features (X) e a variável alvo (y)
X = df[['Bedroom', 'Bathroom', 'DateSold', 'YearBuilt']]  # Features
y = df['Price']  # Variável alvo

# Dividir os dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliar o modelo na validação
y_val_pred = model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
print(f'MSE na validação: {mse_val}')

# Avaliar o modelo no teste
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE no teste: {mse_test}')
