from protocolo_teste import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

df = pd.read_csv('normalized.csv')

# Análise Gráfica dos Resultados:

# Resíduos para regressão linear
residuals_lr = YTest - PrevYRl

plt.figure(figsize=(10, 6))
sns.scatterplot(x=PrevYRl, y=residuals_lr)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (Linear Regression)')
plt.show()

# Importância do recurso da Random Forest
importance = pd.DataFrame(ModeloRf.feature_importances_,
                                   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance['importance'], y=importance.index)
plt.title("Feature Importances in Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Resíduos para Floresta Aleatória
residuals_rf = YTest - PrevYRl

plt.figure(figsize=(10, 6))
sns.scatterplot(x=PrevYRl, y=residuals_rf)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (Random Forest)')
plt.show()

# Atual vs Previsto para a Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(YTest, PrevYRf, alpha=0.5)
plt.plot([YTest.min(), YTest.max()], [YTest.min(), YTest.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (Random Forest)')
plt.show()

# Distribuição de Residuos para a Random Forest
plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, kde=True, bins=30)
plt.title('Distribution of Residuals (Random Forest)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Curva de aprendizagem para a regressão linear
plot_learning_curve(ModeloRl, "Learning Curve (Linear Regression)", XTrain, YTrain, cv=5)
plt.show()

# Curva de aprendizagem para a Random Forest
#plot_learning_curve(rf_model, "Learning Curve (Random Forest)", x_train, y_train, cv=5)
#plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()