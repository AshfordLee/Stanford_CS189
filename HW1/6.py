import numpy as np
from sklearn import svm
import random

def evaluation(true_labels,predicted_labels):
    tp=0

    for true_label,predicted_label in zip(true_labels,predicted_labels):
        if true_label==predicted_label:
            tp+=1

    total=len(true_labels)

    return tp/total

def k_fold_cross_validation(data,labels,k=5,C=1):
    n_samples=len(data)
    indices=list(range(n_samples))
    random.shuffle(indices)

    fold_size=n_samples//k
    accuracies=[]

    for i in range(k):
        start_idx=i*fold_size
        end_idx=start_idx+fold_size if i!=k-1 else n_samples

        val_indices=indices[start_idx:end_idx]
        train_indices=indices[:start_idx]+indices[end_idx:]

        train_data=data[train_indices]
        train_labels=labels[train_indices]

        val_data=data[val_indices]
        val_labels=labels[val_indices]

        train_data=train_data.reshape(train_data.shape[0],-1)
        val_data=val_data.reshape(val_data.shape[0],-1)

        model=svm.SVC(C=C,kernel='linear',random_state=42)
        model.fit(train_data,train_labels)

        val_predictions=model.predict(val_data)

        accuracy=evaluation(val_labels,val_predictions)

        accuracies.append(accuracy)

    return accuracies
        

def find_best_spam():
    data=np.load("./data/spam-data.npz")
    X=data['training_data'][:2000]
    y=data['training_labels'][:2000]

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    results = {}

    for C in C_values:
        accuracies=k_fold_cross_validation(X,y,C=C)
        results[C]=np.mean(accuracies)

    best_C=max(results,key=results.get)
    best_accuracy=results[best_C]

    print(f"The best C is {best_C} with accuracy {best_accuracy}")

    return best_C,best_accuracy


if __name__=="__main__":
    find_best_spam()