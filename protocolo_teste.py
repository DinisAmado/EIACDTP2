import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#Criação do protocolo de treino e avaliação
FinalDf = pd.read_csv('normalized.csv', low_memory=False)
XTrain, XTest, YTrain, YTest = train_test_split(FinalDf.drop(columns=['Price']), FinalDf['Price'], test_size=0.2, random_state=42)
                                                    
print("Dataset size: {0}".format(len(FinalDf)))
print("Training dataset size: {0}".format(len(XTrain)))
print("Test dataset size: {0}".format(len(XTest)))

# Treinar o modelo de regressão linear
ModeloRl = LinearRegression()
ModeloRl.fit(XTrain, YTrain)

# Previsão ao usar o modelo de regressão linear
PrevYRl = ModeloRl.predict(XTest)

# Avaliação do modelo de regressão linear
RlMse = mean_squared_error(YTest, PrevYRl)
AvalRl = r2_score(YTest, PrevYRl)

print("Linear Regression - Mean Squared Error: ", RlMse)
print("Linear Regression - R2 Score: ", AvalRl)

# Treinar o modelo Random Forest Regressor
ModeloRf = RandomForestRegressor(random_state=42)
ModeloRf.fit(XTrain, YTrain)

# Previsão ao usar o modelo Random Forest Regressor
PrevYRf = ModeloRf.predict(XTest)

# Avaliação do modelo Random Forest Regressor
RfMse = mean_squared_error(YTest, PrevYRf)
AvalRf = r2_score(YTest, PrevYRf)

print("Random Forest Regressor - Mean Squared Error: ", RfMse)
print("Random Forest Regressor - R2 Score: ", AvalRf)

# Mean Absolute Error
MaeRl = mean_absolute_error(YTest, PrevYRl)
MaeRf = mean_absolute_error(YTest, PrevYRf)

print("Linear Regression - Mean Absolute Error: ", MaeRl)
print("Random Forest Regressor - Mean Absolute Error: ", MaeRf)

# Cross-Validation for Random Forest Regressor
cv_scores_rf = cross_val_score(ModeloRf, FinalDf.drop(columns=['Price']), FinalDf['Price'], cv=5, scoring='r2')
print("Random Forest Regressor - Cross-Validation R2 Scores: ", cv_scores_rf)
print("Random Forest Regressor - Average Cross-Validation R2 Score: ", cv_scores_rf.mean())

# Cross-Validation for Linear Regression
cv_scores_lr = cross_val_score(ModeloRf, FinalDf.drop(columns=['Price']), FinalDf['Price'], cv=5, scoring='r2')
print("Linear Regression - Cross-Validation R2 Scores: ", cv_scores_lr)
print("Linear Regression - Average Cross-Validation R2 Score: ", cv_scores_lr.mean())