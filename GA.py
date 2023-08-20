from abc import abstractmethod
import random
import sys
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import time


# 忽略特定警告
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

class Algorithm():
    @abstractmethod
    def result(self):
        pass
class GeneticAlgorithm(Algorithm):
    def __init__(self,layer1, layer2, layer3, epochs):
        
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.epochs = epochs

        self.l1_maxsize = 511
        self.l2_maxsize = 511
        self.l3_maxsize = 511
        self.epochs_maxsize = 7

        self.num_parents_to_keep_rate = 0.5

        self.mutaterate = 0.1
        self.init_population_size = 2
        self.generation = 20
        self.data=pd.read_csv('20230817_0050.csv')

        self.graph_data = [] # 初始化列表
    def show_graph(self):
        plt.plot(self.graph_data, marker='o')  # 使用 marker 参数添加数据点标记
        plt.xlabel("X-axis Generation")
        plt.ylabel("Y-axis MSE")
        plt.title("Genetic Algorithm")

        plt.show()
    def result(self):
        # 初始化族群
        population = self.InitPopulation()

        for gen in range(self.generation):
            print(f"Generation {gen + 1} / {self.generation}")

            # 计算每个个体的适应度，并根据适应度选择下一代个体
            selected_population = self.SelectPopulation(population)
            print(f"Selected population {selected_population}")
            
            self.graph_data.append(selected_population[0]['fitness'])

            # 交叉和变异生成新的个体
            new_population = self.CrossOverAndMutate(selected_population)

            # 更新族群
            population = new_population
        population = self.SelectPopulation(population)
        # 从最终的族群中选择最优的个体
        best_individual = population[0]
        self.graph_data.append(best_individual['fitness'])
        print(best_individual)
        # 解码最优个体的超参数并返回结果
        layer1, layer2, layer3, epochs= self.Decode(best_individual['encoded_string'])

        self.show_graph()
        
        return {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'epochs': epochs
        }
        
    def Encode(self,layer1, layer2, layer3, epochs):
        encoded_parameters = []

        # 将 layer1、layer2、layer3、epochs 转换为对应的二进制字串
        encoded_parameters.append(format(layer1, '09b'))  # 使用 9 位的二进制表示，因为 512 的二进制表示最多 9 位
        encoded_parameters.append(format(layer2, '09b'))
        encoded_parameters.append(format(layer3, '09b'))
        encoded_parameters.append(format(epochs, '03b'))

        # 将所有部分的二进制字串连接起来
        return ''.join(encoded_parameters)

    def Decode(self,encoded_string):
        # 在这个方法中，你需要从编码的二进制字串中解析出各个参数的值

        # 解析 layer1、layer2、layer3、epochs 的部分
        layer1_binary = encoded_string[:9]
        layer2_binary = encoded_string[9:18]
        layer3_binary = encoded_string[18:27]
        epochs_binary = encoded_string[27:30]

        # 将二进制字串解析为对应的整数值
        layer1 = int(layer1_binary, 2)
        layer2 = int(layer2_binary, 2)
        layer3 = int(layer3_binary, 2)
        epochs = int(epochs_binary, 2)
        return layer1, layer2, layer3, epochs

        
    def InitPopulation(self):
        population = []
        initdata = self.Encode(self.layer1, self.layer2, self.layer3, self.epochs)
        population.append({"encoded_string":initdata,"fitness":sys.maxsize})

        for _ in range(self.init_population_size-1):
            layer1 = random.randint(16, self.l1_maxsize)
            layer2 = random.randint(16, self.l2_maxsize)
            layer3 = random.randint(16, self.l3_maxsize)
            epochs = random.randint(1,self.epochs_maxsize)


            encoded_string = self.Encode(layer1, layer2, layer3, epochs)
            fitness = sys.maxsize

            individual = {
                "encoded_string": encoded_string,
                "fitness": fitness
            }

            population.append(individual)

        return population
    
    
    def SelectSubsetPopulation(self,subset_population):
            for individual in subset_population:
                if individual['fitness'] == sys.maxsize:
                    individual['fitness'] = self.Fitness(individual['encoded_string'])
            return subset_population

    def SelectPopulation(self, population):
        print('population:',population)
        num_processes = multiprocessing.cpu_count()  # 獲取 CPU 核心數

        if len(population) <= num_processes:
            # 如果個體數小於等於 CPU 核心數，則將整個 population 視為一個子集
            selected_population = self.SelectSubsetPopulation(population)
        else:
            pool = multiprocessing.Pool(processes=num_processes)

            # 分割 population 爲多個子集，每個子集由一個進程處理
            chunk_size = len(population) // num_processes
            remainder = len(population) % num_processes
            subsets = []
            for i in range(num_processes):
                subsets.append([])
            for i in range(chunk_size):
                for j in range(num_processes):
                    index = i * num_processes + j
                    if index < len(population):
                        subsets[j].append(population[index])
             # 将余数均匀分配到子集中
            for i in range(remainder):
                index = chunk_size * num_processes + i
                subsets[i].append(population[index])
            print('subsets:',subsets)

            # 使用多進程處理子集
            selected_subsets = pool.map(self.SelectSubsetPopulation, subsets)
            # 关闭进程池
            pool.close()

            # 等待所有子进程完成
            pool.join()

            # 合併選擇的子集
            selected_population = [item for sublist in selected_subsets for item in sublist]
            # 对 selected_population 进行排序
        
        selected_population.sort(key=lambda ind: ind['fitness'])
        print('selected population:',selected_population)
        selected_population = selected_population[:self.init_population_size ]
        return selected_population
    
    def EnforceParameterRange(self, encoded_string):
        layer1, layer2, layer3, epochs = self.Decode(encoded_string)

        # 对超参数进行范围限制
        layer1 = random.randint(16, self.l1_maxsize)
        layer2 = random.randint(16, self.l2_maxsize)
        layer3 = random.randint(16, self.l3_maxsize)
        epochs = random.randint(1,self.epochs_maxsize)
        # 重新编码为二进制字符串
        new_encoded_string = self.Encode(layer1, layer2, layer3, epochs)
        return new_encoded_string

    def CrossOverAndMutate(self, selected_population):
        new_population = []

        # 计算需要保留的父代数量
        num_parents_to_keep = int(self.init_population_size * self.num_parents_to_keep_rate)

        for _ in range(self.init_population_size):
            # 随机选择父代个体
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            crossover_point = random.randint(0, len(parent1['encoded_string']) - 1)
            child1_encoded = parent2['encoded_string'][:crossover_point] + parent1['encoded_string'][crossover_point:]
            child2_encoded = parent1['encoded_string'][:crossover_point] + parent2['encoded_string'][crossover_point:]
            # 变异操作，随机改变一些基因
            mutated1_encoded = self.Mutate(child1_encoded)
            mutated2_encoded = self.Mutate(child2_encoded)
            # 对变异后的基因进行范围限制
            mutated1_encoded = self.EnforceParameterRange(mutated1_encoded)
            mutated2_encoded = self.EnforceParameterRange(mutated2_encoded)

            new_individual1 = {
                "encoded_string": mutated1_encoded,
                "fitness": sys.maxsize  # 适应度需要重新评估
            }
            new_individual2 = {
                "encoded_string": mutated2_encoded,
                "fitness": sys.maxsize  # 适应度需要重新评估
            }

            new_population.append(new_individual1)
            new_population.append(new_individual2)
        # 保留父代的一部分
        new_population.extend(selected_population[:num_parents_to_keep])

        return new_population
    def Mutate(self, encoded_string):
        mutated_string = list(encoded_string)
        
        for i in range(len(mutated_string)):
            if random.random() < self.mutaterate:
                mutated_string[i] = '0' if mutated_string[i] == '1' else '1'

        return ''.join(mutated_string)
    
    def Fitness(self, encoded_string):
        layer1,layer2,layer3,epochs = self.Decode(encoded_string)
        # 获取数据并转换为DataFrame格式
        data = self.data
        # 获取时间戳（以秒为单位）
        data['date'] = pd.to_datetime(data['date'])
        data['timestamp'] = data['date'].apply(lambda x: x.timestamp())

        # 添加meanprice特征
        data['meanprice'] = (data['highprice'] + data['lowprice']) / 2

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

        data = {'layer1':layer1,'layer2':layer2,'layer3':layer3,'epochs':epochs}

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
        y_pred = scaler.inverse_transform(y_pred_scaled)

        # 将 y_test 转换为原始范围
        y_test_original = scaler.inverse_transform(y_test)

        # 计算均方误差
        mse = mean_squared_error(y_test_original, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

def main():
    # 记录开始时间
    start_time = time.time()

    ga = GeneticAlgorithm(layer1=32, layer2=32, layer3=32,epochs=7)
    # encodestr = ga.Encode(layer1=32, layer2=32, layer3=32,epochs=7)
    # decodestr = ga.Decode(encodestr)
    # print(encodestr)
    # print(decodestr)
    result=ga.result()
    print(result)
    # 记录结束时间
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    print(f"execution time: %.2f seconds" %execution_time)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

