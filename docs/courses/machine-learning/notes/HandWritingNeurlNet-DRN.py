import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy
import sklearn
import time
from datetime import timedelta
import pickle
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog

print("NumPy版本:", numpy.__version__)
print("Scikit-learn版本:", sklearn.__version__)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # 初始化网络参数
        # 如果初始权重过大，输入到激活函数（如 sigmoid）的值可能会非常大或非常小，
        # 导致激活函数的输出接近其极限值（0 或 1）。这会使得梯度接近于零，从而导致梯度消失问题，阻碍网络的学习
        # 从标准正态分布（均值为 0，标准差为 1）中随机采样的浮点数
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.1
        self.bias1 = np.zeros((1, hidden_size1))
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        self.bias2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.1
        self.bias3 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # 前向传播
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        # 输出层用的是softmax，公式是a=e^z/sum(e^z)
        self.output = self.softmax(np.dot(self.layer2, self.weights3) + self.bias3)
        return self.output

    def backward(self, X, y, learning_rate):
        batch_size = X.shape[0]
        
        # 输出层误差 δ^(L) = ∂C/∂a^(L)
        # 对于softmax输出层的简化形式，公式是δ=a-y
        delta3 = self.output - y  # 对于softmax输出层的简化形式
        
        # 隐藏层2误差 δ^(L-1) = ((w^(L))^T * δ^(L)) * σ'(z^(L-1))
        delta2 = np.dot(delta3, self.weights3.T) * self.sigmoid_derivative(self.layer2)
        
        # 隐藏层1误差 δ^(L-2) = ((w^(L-1))^T * δ^(L-1)) * σ'(z^(L-2))
        delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_derivative(self.layer1)

        # 权重更新公式：w^(L) = w^(L) - η * δ^(L) * (a^(L-1))^T,使用 mini-batch 梯度下降
        self.weights3 -= learning_rate * np.dot(self.layer2.T, delta3) / batch_size
        self.weights2 -= learning_rate * np.dot(self.layer1.T, delta2) / batch_size
        self.weights1 -= learning_rate * np.dot(X.T, delta1) / batch_size
        # np.dot(self.layer2.T, delta3)返回的是 累加的梯度总和，不是平均值。
        # 而标准的梯度下降更新，其实是想用 每个样本的平均梯度，让不同 batch size 的训练过程有更一致的步长

        # 偏置更新公式：b^(L) = b^(L) - η * δ^(L)
        self.bias3 -= learning_rate * np.sum(delta3, axis=0, keepdims=True) / batch_size
        self.bias2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True) / batch_size
        self.bias1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True) / batch_size

    def save_model(self, filename):
        """保存模型参数"""
        model_params = {
            'weights1': self.weights1,
            'weights2': self.weights2,
            'weights3': self.weights3,
            'bias1': self.bias1,
            'bias2': self.bias2,
            'bias3': self.bias3
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filename):
        """加载模型参数"""
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        self.weights1 = model_params['weights1']
        self.weights2 = model_params['weights2']
        self.weights3 = model_params['weights3']
        self.bias1 = model_params['bias1']
        self.bias2 = model_params['bias2']
        self.bias3 = model_params['bias3']
    
    def predict(self, X):
        """预测单个图像"""
        # 确保输入是2D数组
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        # 归一化，将每个像素的值归一化到 [0, 1] 的范围内
        X = X / 255.0
        # 前向传播
        output = self.forward(X)
        # 返回预测的数字
        return np.argmax(output, axis=1)[0]

def preprocess_data():
    # 加载MNIST数据集
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # 数据归一化
    X = X / 255.0
    
    # 将标签转换为one-hot编码
    y_onehot = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        y_onehot[i, int(y[i])] = 1
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    # 初始化参数
    input_size = 784
    hidden_size1 = 30
    hidden_size2 = 60
    output_size = 10
    learning_rate = 0.1
    epochs = 50
    batch_size = 128

    # 加载和预处理数据
    X_train, X_test, y_train, y_test = preprocess_data()

    # 创建神经网络
    nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

    print("开始训练...")
    start_time = time.time()
    total_batches = len(X_train) // batch_size

    # 训练网络
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 随机打乱训练数据
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # 批量训练
        for i in range(0, len(X_train), batch_size):
            if i % (batch_size * 20) == 0:
                print(f'\rEpoch {epoch + 1}/{epochs} - 批次进度: {i//batch_size}/{total_batches}', end='')
            
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            nn.forward(X_batch)
            nn.backward(X_batch, y_batch, learning_rate)

        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 5 == 0:
            train_predictions = np.argmax(nn.forward(X_train), axis=1)
            train_true = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(train_predictions == train_true)

            test_predictions = np.argmax(nn.forward(X_test), axis=1)
            test_true = np.argmax(y_test, axis=1)
            test_accuracy = np.mean(test_predictions == test_true)

            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'每轮用时: {timedelta(seconds=int(epoch_time))}')
            print(f'训练集准确率: {train_accuracy:.4f}')
            print(f'测试集准确率: {test_accuracy:.4f}')

    total_time = time.time() - start_time
    print(f'\n训练完成！总用时: {timedelta(seconds=int(total_time))}')

    # 在训练完成后保存模型
    print("保存模型...")
    nn.save_model('mnist_model.pkl')

# 添加新的函数用于预测
def predict_digit(image_data):
    """
    预测单个手写数字
    image_data: 784维的numpy数组（28x28像素）
    """
    # 创建模型实例
    nn = NeuralNetwork(784, 30, 60, 10)
    # 加载训练好的模型
    nn.load_model('mnist_model.pkl')
    # 进行预测
    return nn.predict(image_data)

def load_image(image_path):
    """
    加载并处理图片文件
    image_path: 图片文件的路径
    返回: 处理后的图片数据（784维numpy数组）
    """
    # 打开图片
    img = Image.open(image_path)
    # 转换为灰度图
    img = img.convert('L')
    # 调整大小为28x28像素
    img = img.resize((28, 28))
    # 转换为numpy数组并归一化
    img_array = np.array(img)
    # 反转颜色（确保白底黑字）
    img_array = 255 - img_array
    # 展平为一维数组
    img_array = img_array.reshape(784)
    return img_array

def select_image():
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保对话框在最前面
        
        file_path = filedialog.askopenfilename(
            title='选择手写数字图片',
            filetypes=[
                ('图片文件', '*.png *.jpg *.jpeg *.bmp *.gif'),
                ('所有文件', '*.*')
            ]
        )
        
        root.destroy()  # 确保完全关闭Tk窗口
        return file_path if file_path else None
        
    except Exception as e:
        print(f"选择文件时出错：{str(e)}")
        return None

def select_image_alternative():
    """备选的文件选择方法"""
    print("\n请直接输入图片文件的完整路径：")
    file_path = input().strip()
    if os.path.exists(file_path):
        return file_path
    print("文件不存在！")
    return None

# 使用示例
if __name__ == "__main__":
    if input("是否需要重新训练模型？(y/n): ").lower() == 'y':
        main()
    else:
        if not os.path.exists('mnist_model.pkl'):
            print("\n错误：找不到模型文件 'mnist_model.pkl'")
            print("请先训练模型（输入 'y' 进行训练）再进行预测。")
            exit()
            
        try:
            print("\n选择预测模式：")
            print("1. 使用MNIST测试集样本")
            print("2. 使用本地图片文件")
            choice = input("请选择（1/2）：")
            
            nn = NeuralNetwork(784, 30, 60, 10)
            nn.load_model('mnist_model.pkl')
            
            if choice == '1':
                # 使用MNIST测试集
                print("\n加载测试数据...")
                X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
                test_samples = X[:10]
                true_labels = y[:10]
                
                print("\n预测演示：")
                for i, (sample, true_label) in enumerate(zip(test_samples, true_labels)):
                    predicted = nn.predict(sample)
                    print(f"样本 {i+1}: 预测值 = {predicted}, 实际值 = {true_label}")
            
            elif choice == '2':
                print("\n准备打开文件选择对话框...")
                try:
                    print("\n请在弹出的对话框中选择图片文件...")
                    image_path = select_image()
                    if not image_path:
                        print("\n使用备选方法...")
                        image_path = select_image_alternative()
                    if not image_path:
                        print("未能获取有效的图片文件路径")
                        exit()
                    
                    print(f"已选择文件：{image_path}")
                    img_data = load_image(image_path)
                    predicted = nn.predict(img_data)
                    print(f"\n预测结果：这个数字是 {predicted}")
                except Exception as e:
                    print(f"\n选择文件过程中发生错误：{str(e)}")
                    print("请确保系统支持文件对话框操作。")
            
            else:
                print("无效的选择")
                
        except Exception as e:
            print(f"\n发生错误：{str(e)}")
            print("请确保已经完成模型训练，并且模型文件正确保存。")
