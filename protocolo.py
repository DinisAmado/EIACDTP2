import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Passo 1: Carregar os dados normalizados
df_normalized2 = pd.read_csv('csv_normalized2.csv')

# Passo 2: Definir as features (X) e a variável alvo (y)
features = ['Bedroom', 'Bathroom', 'Area']
X = df_normalized2[features]
y = df_normalized2['Price']

# Passo 3: Dividir os dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Passo 4: Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Passo 5: Avaliar o modelo na validação
y_val_pred = model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
print(f'MSE na validação: {mse_val}')

# Avaliar o modelo no teste
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE no teste: {mse_test}')

# Calcular o R² para os dados de validação
r2_val = r2_score(y_val, y_val_pred)
print(f'R² na validação: {r2_val}')

# Calcular o R² para os dados de teste
r2_test = r2_score(y_test, y_test_pred)
print(f'R² no teste: {r2_test}')

# Passo 6: Visualizar os resultados
# Gráfico de previsões vs valores reais (validação)
plt.figure(figsize=(10, 5))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs Valores Reais (Validação)')
plt.show()

# Gráfico de previsões vs valores reais (teste)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs Valores Reais (Teste)')
plt.show()

# Gráfico de resíduos (validação)
residuos_val = y_val - y_val_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_val_pred, residuos_val, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Previsões')
plt.ylabel('Reais')
plt.title('Reais vs Previsões (Validação)')
plt.show()

# Gráfico de resíduos (teste)
residuos_test = y_test - y_test_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_test_pred, residuos_test, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Previsões')
plt.ylabel('Reais')
plt.title('Reais vs Previsões (Teste)')
plt.show()
