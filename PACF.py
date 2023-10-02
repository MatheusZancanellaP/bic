import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Carregar o conjunto de dados imputado
df = pd.read_csv("global_irradiance_daily_aurora_imputed.csv", delimiter=";")
df['Data'] = pd.to_datetime(df['Data'])

# Ordenar os dados pela coluna de data
df = df.sort_values('Data')

# Selecionar a coluna de medição
medicao = df['Medicao']

# Calcular e plotar o PACF
plot_pacf(medicao, lags=100)  # Você pode ajustar o número de lags conforme necessário
plt.show()