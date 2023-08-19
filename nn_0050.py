import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def lgbm0050():
    # 获取数据并转换为DataFrame格式
    data = pd.read_csv('20230817_0050.csv')
    # 获取时间戳（以秒为单位）
    data['date'] = pd.to_datetime(data['date'])
    data['timestamp'] = data['date'].apply(lambda x: x.timestamp())

    # 添加meanprice特征
    data['meanprice'] = (data['highprice'] + data['lowprice']) / 2

    data.to_csv('20230817_0050.csv')

    # 特征和标签分离
    X = data.drop(['meanprice','date', 'close'], axis=1)
    y = data['close']

    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

   # 找到分割索引，将数据分为训练集和测试集
    split_index = int(len(X_scaled) * 0.8)  # 80% 的数据用于训练，20% 的数据用于测试

    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]


    # 转换数据形状以适应LSTM模型 (样本数, 时间步数, 特征数)
    n_steps = 1  # 假设每个样本有1个时间步
    n_features = X_train.shape[1]
    X_train_reshaped = X_train.reshape((X_train.shape[0], n_steps, n_features))
    X_test_reshaped = X_test.reshape((X_test.shape[0], n_steps, n_features))

    data = {'layer1': 259, 'layer2': 410, 'layer3': 473, 'epochs': 7}

    # 定义LSTM模型
    model = Sequential()
    model.add(Bidirectional(LSTM(data['layer1'], activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
    model.add(Bidirectional(LSTM(data['layer2'], activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(data['layer3'], activation='relu',return_sequences=False)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train_reshaped, y_train, epochs=data['epochs'], batch_size=128)

    y_pred_scaled = model.predict(X_test_reshaped)
    print(y_pred_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # 将 y_test 转换为原始范围
    y_test_original = scaler.inverse_transform(y_test)

    # 计算均方误差
    mse = mean_squared_error(y_test_original, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    return mse

if __name__ == '__main__':
    print(lgbm0050())