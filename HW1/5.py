import numpy as np
from sklearn import svm

def partition_MNIST():
    data=np.load('./data/mnist-data.npz')
    train_data=data['training_data'][10000:30000]
    train_labels=data['training_labels'][10000:30000]
    validation_data=data['training_data'][:10000]
    validation_labels=data['training_labels'][:10000]
    return train_data,train_labels,validation_data,validation_labels


def evaluation(true_labels,predicted_labels):
    tp=0

    for true_label,predicted_label in zip(true_labels,predicted_labels):
        if true_label==predicted_label:
            tp+=1

    total=len(true_labels)

    return tp/total

def train_model(train_data,train_labels,valid_data,valid_labels,C):
    train_data=train_data.reshape(train_data.shape[0],-1)
    valid_data=valid_data.reshape(valid_data.shape[0],-1)

    model=svm.SVC(C=C,kernel='linear')
    print(f"Starting training model with parameter C={C}")
    model.fit(train_data,train_labels)

    accuracy=evaluation(valid_labels,model.predict(valid_data))
    print(f"Accuracy of parameter C={C} is {accuracy}")

    return accuracy,model

def train_model_full(train_data,train_labels,C):
    train_data=train_data.reshape(train_data.shape[0],-1)

    model=svm.SVC(C=C,kernel='linear')
    print(f"Starting training model with parameter C={C}")
    model.fit(train_data,train_labels)

    accuracy=evaluation(train_labels,model.predict(train_data))
    print(f"Accuracy of parameter C={C} is {accuracy}")

    return model

def main():
    train_data,train_labels,validation_data,validation_labels=partition_MNIST()
    acc={}
    C_values = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    for C in C_values:
    # C=0.1

        accuracy,model=train_model(train_data,train_labels,validation_data,validation_labels,C)
        acc[model]=[accuracy,C]

    max_acc=sorted(acc.items(),key=lambda x:x[1],reverse=True)[0]
    print(f"The best model is {max_acc[0]} with accuracy {max_acc[1][0]} and C={max_acc[1][1]}")

    model=train_model_full(train_data,train_labels,max_acc[1][1])


    
if __name__ == "__main__":
    main()

