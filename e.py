import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import keras_tuner as kt
from kerastuner.tuners import GridSearch
from statsmodels.tsa.stattools import adfuller
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


# Função para calcular o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Carregue os dados do arquivo CSV
df = pd.read_csv('global_irradiance_daily_aurora.csv', delimiter=';', parse_dates=['Data'])

# Trate os dados faltantes removendo as linhas com NaNs
df['Medicao'].replace('', np.nan, inplace=True)
df.dropna(subset=['Medicao'], inplace=True)

# Selecione apenas a coluna "Medicao"
data = df['Medicao'].values.reshape(-1, 1)

# Plotagem da série original e após remoção de outliers lado a lado
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Série Original
axes[0].plot(data, label='Série Original')
axes[0].set_title('Série Original')
axes[0].set_xlabel('Dias')
axes[0].set_ylabel('Medição')
axes[0].legend()

# Remoção de outliers usando Z-Score
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_filtered = data[filtered_entries]

# Série após remoção de outliers
axes[1].plot(data_filtered, label='Série sem Outliers', color='orange')
axes[1].set_title('Série após Remoção de Outliers')
axes[1].set_xlabel('Dias')
axes[1].set_ylabel('Medição')
axes[1].legend()

number_of_outliers = len(data) - len(data_filtered)
print("outliers numbers:" + str(number_of_outliers))

plt.tight_layout()
plt.show()

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in result[4].items():
        output['Critical Value (%s)' % key] = value
    print(output)
    if output['p-value'] <= 0.05:
        print("A série é estacionária.")
    else:
        print("A série não é estacionária.")

# Realize o teste ADF na série após a remoção dos outliers
print("\nResultado do teste Augmented Dickey-Fuller:")
adf_test(data_filtered.ravel())


# Normalize os dados
scaler = MinMaxScaler(feature_range=(0, 1))
data_filtered = scaler.fit_transform(data_filtered)

# Divida os dados em conjuntos de treinamento e teste
train_size = int(len(data_filtered) * 0.8)
train, test = data_filtered[0:train_size,:], data_filtered[train_size:len(data_filtered),:]

# Converta os dados para o formato [amostras, passos de tempo, características]
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Função para construir o modelo com hiperparâmetros variáveis
def build_model(hp):
    model = Sequential()
    
    # Camada LSTM
    model.add(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=2048, step=32),
                   input_shape=(1, look_back)))
    
    # Dropout para regularização
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Camadas Densas adicionais
    for i in range(hp.Int('n_layers', 1, 4)):  # número de camadas densas
        model.add(Dense(units=hp.Int(f'dense_units_{i}', min_value=32, max_value=2048, step=32),
                        activation='sigmoid'))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Camada de saída
    model.add(Dense(1))
    
    # Otimizador e taxa de aprendizado
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, sampling='LOG'))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Configuração do KerasTuner com GridSearch
tuner = GridSearch(
    build_model,
    objective='val_loss',
    max_trials=30,
    directory='output_dir',
    project_name='lstm_gridsearch'
)

# Busca pelos melhores hiperparâmetros
tuner.search(trainX, trainY, epochs=100, validation_data=(testX, testY))

# Obtenha o melhor modelo e hiperparâmetros
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Avaliação do Modelo
trainPredict = best_model.predict(trainX)
testPredict = best_model.predict(testX)

# Inverta as previsões para a escala original
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calcule o MAPE
trainScore = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore))

# Plotagem das séries
plt.figure(figsize=(15, 6))
plt.plot(testY[0], label='Valores Reais')
plt.plot(testPredict, label='Previsões LSTM')
plt.title('Comparação entre Valores Reais e Previsões LSTM')
plt.xlabel('Dias')
plt.ylabel('Medição')
plt.legend()
plt.show()
