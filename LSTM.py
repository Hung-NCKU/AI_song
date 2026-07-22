import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成一些示例数据（用您自己的数据集替换这部分）
def generate_time_series():
    time = np.arange(0, 100, 0.1)
    sin_wave = np.sin(time)
    return sin_wave

def create_dataset(series, window_size):
    data = []
    labels = []

    for i in range(len(series) - window_size):
        window = series[i:i+window_size]
        target = series[i+window_size]
        data.append(window)
        labels.append(target)

    return np.array(data), np.array(labels)

# 生成并预处理数据
time_series = generate_time_series()
window_size = 10
X, y = create_dataset(time_series, window_size)

# 为LSTM输入重塑数据（样本，时间步长，特征）
X = X.reshape((X.shape[0], window_size, 1))

# 构建LSTM模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X, y, epochs=100, batch_size=32)

# 进行预测
test_series = generate_time_series()
X_test, y_test = create_dataset(test_series, window_size)
X_test = X_test.reshape((X_test.shape[0], window_size, 1))
predictions = model.predict(X_test)

# 评估模型
mse = np.mean((predictions - y_test)**2)
print(f"均方误差：{mse}")