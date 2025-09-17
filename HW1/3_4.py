import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def partition_MNIST():
    data=np.load('./data/mnist-data.npz')
    train_data=data['training_data'][10000:]
    train_labels=data['training_labels'][10000:]
    validation_data=data['training_data'][:10000]
    validation_labels=data['training_labels'][:10000]
    return train_data,train_labels,validation_data,validation_labels

def partition_spam():
    data=np.load('./data/spam-data.npz')
    total_samples = len(data['training_data'])
    validation_size = int(total_samples * 0.2)

    indices = np.random.permutation(total_samples)
    train_indices = indices[validation_size:]
    val_indices = indices[:validation_size]


    train_data=data['training_data'][train_indices]
    train_labels=data['training_labels'][train_indices]
    validation_data=data['training_data'][val_indices]
    validation_labels=data['training_labels'][val_indices]

    return train_data,train_labels,validation_data,validation_labels

def evaluation(true_labels,predicted_labels):
    tp=0

    for true_label,predicted_label in zip(true_labels,predicted_labels):
        if true_label==predicted_label:
            tp+=1

    total=len(true_labels)

    return tp/total

def mnist_svm(train_data,train_labels):
    train_data=train_data.reshape(train_data.shape[0],-1)
    clf=svm.SVC(kernel='linear')
    clf.fit(train_data,train_labels)

    return clf

def evaluate_accuracy(model,data,labels):
    data=data.reshape(data.shape[0],-1)
    predictions=model.predict(data)
    return evaluation(labels,predictions)

def train_and_evaluate_svm(train_data,train_labels,validation_data,validation_labels,training_sizes):
    train_accuracies=[]
    val_accuracies=[]

    for size in training_sizes:
        train_subset=train_data[:size]
        train_labels_subset=train_labels[:size]

        model=mnist_svm(train_subset,train_labels_subset)

        train_acc=evaluate_accuracy(model,train_subset,train_labels_subset)
        val_acc=evaluate_accuracy(model,validation_data,validation_labels)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return train_accuracies,val_accuracies

def plot_learning_curve(training_sizes, train_accs, val_accs, dataset_name="MNIST"):
    """绘制学习曲线：训练准确率和验证准确率 vs 训练样本数"""
    
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(training_sizes, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(training_sizes, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    
    # 设置图形属性
    plt.xlabel('Number of Training Examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Learning Curve: {dataset_name} Dataset', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # 对数刻度
    plt.ylim(0, 1)
    
    # 添加数值标签
    for size, train_acc, val_acc in zip(training_sizes, train_accs, val_accs):
        plt.annotate(f'{train_acc:.3f}', (size, train_acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{val_acc:.3f}', (size, val_acc), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{dataset_name.lower()}_learning_curve.png', dpi=300, bbox_inches='tight')

if __name__=="__main__":
    train_data, train_labels, val_data, val_labels = partition_MNIST()
    training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    train_accs, val_accs = train_and_evaluate_svm(train_data, train_labels, val_data, val_labels, training_sizes)
    print("MNIST Results:")
    for size, train_acc, val_acc in zip(training_sizes, train_accs, val_accs):
        print(f"Size: {size:5d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")

    plot_learning_curve(training_sizes, train_accs, val_accs, "MNIST")



    train_data, train_labels, val_data, val_labels = partition_spam()
    training_sizes = [100, 200, 500, 1000, 2000,train_data.shape[0]]
    train_accs, val_accs = train_and_evaluate_svm(train_data, train_labels, val_data, val_labels, training_sizes)
    print("Spam Results:")
    for size, train_acc, val_acc in zip(training_sizes, train_accs, val_accs):
        print(f"Size: {size:5d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}")

    plot_learning_curve(training_sizes, train_accs, val_accs, "Spam")