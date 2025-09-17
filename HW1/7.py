import numpy as np
from sklearn import svm
import time

def load_mnist_test_data():
    data=np.load("./data/mnist-data.npz")
    data=data['test_data']
    return data

def load_spam_test_data():
    data=np.load("./data/spam-data.npz")
    data=data['test_data']
    return data

def load_mnist_train_data():
    data=np.load("./data/mnist-data.npz")
    train_data=data['training_data'][:10000]
    train_label=data['training_labels'][:10000]
    return train_data,train_label

def load_spam_train_data():
    data=np.load("./data/spam-data.npz")
    train_data=data['training_data'][1000:]
    train_label=data['training_labels'][1000:]
    return train_data,train_label

def mnist_fit_model(train_data,train_labels):
    print("开始训练MNIST模型...")
    start_time = time.time()
    
    model=svm.LinearSVC(C=9,random_state=42,max_iter=1000)
    model.fit(train_data,train_labels)
    
    training_time = time.time() - start_time
    print(f"MNIST训练完成，耗时: {training_time:.2f}秒")
    
    return model

def mnist_predict_model(model,test_data):
    print("开始预测MNIST测试数据...")
    start_time = time.time()
    
    predictions = model.predict(test_data)
    
    prediction_time = time.time() - start_time
    print(f"MNIST预测完成，耗时: {prediction_time:.2f}秒")
    
    return predictions

def spam_fit_model(train_data,train_labels):
    print("开始训练垃圾邮件模型...")
    start_time = time.time()
    
    model=svm.LinearSVC(C=9,random_state=42,max_iter=1000)
    model.fit(train_data,train_labels)
    
    training_time = time.time() - start_time
    print(f"垃圾邮件训练完成，耗时: {training_time:.2f}秒")
    
    return model

def spam_predict_model(model,test_data):
    print("开始预测垃圾邮件测试数据...")
    start_time = time.time()
    
    predictions = model.predict(test_data)
    
    prediction_time = time.time() - start_time
    print(f"垃圾邮件预测完成，耗时: {prediction_time:.2f}秒")
    
    return predictions

def main():
    print("=== 开始加载数据 ===")
    
    # 加载训练数据
    mnist_train_data,mnist_train_label=load_mnist_train_data()
    spam_train_data,spam_train_label=load_spam_train_data()
    
    print(f"MNIST训练数据形状: {mnist_train_data.shape}")
    print(f"MNIST训练标签形状: {mnist_train_label.shape}")
    print(f"垃圾邮件训练数据形状: {spam_train_data.shape}")
    print(f"垃圾邮件训练标签形状: {spam_train_label.shape}")
    
    # 数据预处理
    print("\n=== 数据预处理 ===")
    mnist_train_data=mnist_train_data.reshape(mnist_train_data.shape[0],-1)
    spam_train_data=spam_train_data.reshape(spam_train_data.shape[0],-1)
    
    print(f"MNIST训练数据reshape后形状: {mnist_train_data.shape}")
    print(f"垃圾邮件训练数据reshape后形状: {spam_train_data.shape}")
    
    # 加载测试数据
    print("\n=== 加载测试数据 ===")
    mnist_test_data=load_mnist_test_data()
    spam_test_data=load_spam_test_data()
    
    # 测试数据也需要reshape
    mnist_test_data=mnist_test_data.reshape(mnist_test_data.shape[0],-1)
    spam_test_data=spam_test_data.reshape(spam_test_data.shape[0],-1)
    
    print(f"MNIST测试数据形状: {mnist_test_data.shape}")
    print(f"垃圾邮件测试数据形状: {spam_test_data.shape}")
    
    # 训练模型
    print("\n=== 开始训练模型 ===")
    mnist_model=mnist_fit_model(mnist_train_data,mnist_train_label)
    spam_model=spam_fit_model(spam_train_data,spam_train_label)
    
    # 预测
    print("\n=== 开始预测 ===")
    mnist_predictions=mnist_predict_model(mnist_model,mnist_test_data)
    spam_predictions=spam_predict_model(spam_model,spam_test_data)
    
    # 计算准确率（修正）
    print("\n=== 结果分析 ===")
    print(f"MNIST预测结果形状: {mnist_predictions.shape}")
    print(f"垃圾邮件预测结果形状: {spam_predictions.shape}")
    
    # 注意：这里需要测试标签来计算准确率
    # 由于您没有测试标签，这里只能显示预测结果
    print(f"MNIST预测结果范围: {np.min(mnist_predictions)} 到 {np.max(mnist_predictions)}")
    print(f"垃圾邮件预测结果范围: {np.min(spam_predictions)} 到 {np.max(spam_predictions)}")
    
    # 保存预测结果
    print("\n=== 保存预测结果 ===")
    np.savetxt("mnist_predictions.txt", mnist_predictions, fmt='%d')
    np.savetxt("spam_predictions.txt", spam_predictions, fmt='%d')
    print("预测结果已保存到文件")

if __name__=="__main__":
    main()