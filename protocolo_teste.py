import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#Criação do protocolo de treino e avaliação

final_df = pd.read_csv('normalized.csv', low_memory=False)
x_train, x_test, y_train, y_test = train_test_split(final_df.drop(columns=['Price']), final_df['Price'], test_size=0.2, random_state=42)
                                                    
print("Tamanho do dataset: {0}".format(len(final_df)))                                                   
print("Tamanho do dataset de treino: {0}".format(len(x_train)))
print("Tamanho do dataset de teste: {0}".format(len(x_test)))

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Predict using the Linear Regression model
y_pred_lr = lr_model.predict(x_test)

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression - Mean Squared Error: ", mse_lr)
print("Linear Regression - R2 Score: ", r2_lr)


# Train Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)

# Predict using the Random Forest Regressor model
y_pred_rf = rf_model.predict(x_test)

# Evaluate Random Forest Regressor model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor - Mean Squared Error: ", mse_rf)
print("Random Forest Regressor - R2 Score: ", r2_rf)


# Mean Absolute Error
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Linear Regression - Mean Absolute Error: ", mae_lr)
print("Random Forest Regressor - Mean Absolute Error: ", mae_rf)


# Cross-Validation for Linear Regression
cv_scores_lr = cross_val_score(lr_model, final_df.drop(columns=['Price']), final_df['Price'], cv=5, scoring='r2')
print("Linear Regression - Cross-Validation R2 Scores: ", cv_scores_lr)
print("Linear Regression - Average Cross-Validation R2 Score: ", cv_scores_lr.mean())

# Cross-Validation for Random Forest Regressor
cv_scores_rf = cross_val_score(rf_model, final_df.drop(columns=['Price']), final_df['Price'], cv=5, scoring='r2')
print("Random Forest Regressor - Cross-Validation R2 Scores: ", cv_scores_rf)
print("Random Forest Regressor - Average Cross-Validation R2 Score: ", cv_scores_rf.mean())

