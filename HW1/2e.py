import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data=np.load("./data/toy-data.npz")

    return data

def plot_data_points(data,labels):
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.show()

def plot_decision_boundary(w=np.array([-0.4528,-0.5190]),b=0.1471):
    x=np.linspace(-5,5,100)
    y=(-w[0]*x-b)/w[1]
    plt.plot(x,y,'k')
    plt.show()

def plot_margin(w=np.array([-0.4528,-0.5190]),b=0.1471):
    x=np.linspace(-5,5,100)
    y=(-w[0]*x-b+1)/w[1]
    plt.plot(x,y,'k')
    y=(-w[0]*x-b-1)/w[1]
    plt.plot(x,y,'k')
    plt.show()
    
if __name__=="__main__":
    data=load_data()
    plot_data_points(data["training_data"],data["training_labels"])
    plot_decision_boundary()
    plot_margin()