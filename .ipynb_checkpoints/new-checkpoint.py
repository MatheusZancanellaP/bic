from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# 1. Carregar o conjunto de dados imputado
df = pd.read_csv("global_irradiance_daily_aurora_imputed.csv", delimiter=";")
df['Data'] = pd.to_datetime(df['Data'])

# 2. Dividir o conjunto de dados em 80% para treinamento e 20% para teste
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
X_train = np.array(range(len(train_df))).reshape(-1, 1)
y_train = train_df['Medicao'].values
X_test = np.array(range(len(train_df), len(train_df) + len(test_df))).reshape(-1, 1)
y_test = test_df['Medicao'].values

# 3. Treinar um modelo usando os dados de treinamento
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Fazer previsões usando o modelo treinado
y_pred = model.predict(X_test)

# 5. Comparar as previsões com os dados de teste usando várias métricas
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Cálculo do MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("MAPE:", mape)
