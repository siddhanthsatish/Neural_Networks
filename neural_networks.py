from cmath import exp
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
# from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split



#stratified k-fold
def create_k_folds(d, k):
    data = pd.read_csv(d)
    dataset_split = []
    df_copy = data
    fold_size = int(df_copy.shape[0] / k)
    vals, counts = np.unique(data['target'], return_counts= True)
    prop = counts/counts.sum()
    sizes = prop*fold_size
    sizes = sizes.astype(int)
    # for loop to save each fold
    for i in range(k):
        fold = []
        # while loop to add elements to the folds
        for ind in range(len(vals)):   
            s = 0
            while s < sizes[ind]:
                # select a random element
                inds =  list(df_copy.where(df_copy['target']==vals[ind]).dropna().index.values)
                index = inds[random.randint(0,len(inds)-1)]
                
                # save the randomly selected line 
                fold.append(df_copy.loc[index].values.tolist())
                
                # delete the randomly selected line from
                # dataframe not to select again
                df_copy = df_copy.drop(index)
                s+=1
        # save the fold     
        test = pd.DataFrame(fold, columns = df_copy.columns.values.tolist())
        train = df_copy
        dataset_split.append([train.to_numpy(), test.to_numpy()])
    return dataset_split

def initialize_network(layer_sizes : list, scale : float):
    init_params = {}
    for layer in range(len(layer_sizes) - 1):
        weight = np.random.randn(layer_sizes[layer], layer_sizes[layer + 1]) * scale
        bias = np.random.randn(layer_sizes[layer + 1])
        init_params["W" + str(layer)] = weight
        init_params["b" + str(layer)] = bias
    return init_params

def sigma_forward(IN: np.ndarray):
    A = 1 / (1 + np.exp(-IN))
    return A

def forward(params: dict, X: np.ndarray):
    cache = {}

    cache["A0"] = np.copy(X)
    cache["IN0"] = np.copy(X)
    loop_count = len(list(params.keys())) // 2
    for i in range(1, loop_count):
        cache["IN" + str(i)] = np.dot(cache["A" + str(i - 1)], params["W" + str(i - 1)]) + params["b" + str(i - 1)]
        print('Z', cache["IN" + str(i)])
        cache["A" + str(i)] = sigma_forward(cache["IN" + str(i)])
        print('A', cache["A" + str(i)])

    cache["IN" + str(loop_count)] = np.dot(cache["A" + str(loop_count - 1)], params["W" + str(loop_count - 1)]) + params["b" + str(loop_count - 1)]
    cache["A" + str(loop_count)] = sigma_forward(cache["IN" + str(loop_count)])
    prediction = cache["A" + str(loop_count)]

    return prediction, cache

def sigma_backward(A: np.ndarray):
    dsigma = (1 - A) * A

    return dsigma


def backprop_and_loss(params: dict, prediction: np.ndarray, cache: dict, Y : np.ndarray,alpha =0):

    gradient = {}

    loop_count = len(params) // 2

    reg = 0
    for param in params:
        if param[0] == "W":
            reg += np.sum(params[param] ** 2)  
    n = Y.shape[0]
    loss  = np.average(np.sum(-Y * np.log(prediction)- (1-Y) * np.log(1-prediction),axis=1)) + alpha * reg / (2*n)
    
    dout = 1/n *  (-Y/prediction + (1-Y)/(1-prediction))
    dout = sigma_backward(cache["A" + str(loop_count)]) * dout
    print('DOUTS')
    for i in range(loop_count):
        index = (loop_count - i) - 1
        dW = np.dot(cache["A" + str(index)].T, dout) 
        db = np.sum(dout,axis=0)
        dx = np.dot(dout, params["W" + str(index)].T)
        gradient["W" + str(index)] = dW + alpha * params["W"+str(index)] / n
        gradient["b" + str(index)] = db
        dout = sigma_backward(cache["A" + str(index)]) * dx
        print(dout)
    return gradient, loss

def gradient_descent(X : np.ndarray, Y : np.ndarray, initial_params : dict, lr : float, num_iterations : int, alpha=0)->Tuple[List[float], np.ndarray]:
    losses = []
    final_params = {}

    for n in range(num_iterations): 
        prediction, cache = forward(initial_params, X)
        gradients, loss = backprop_and_loss(initial_params, prediction, cache, Y,alpha)
        for i in range(len(initial_params) // 2):
            initial_params["W" + str(i)] = initial_params["W" + str(i)] - lr * gradients["W" + str(i)]
        losses.append(loss)
        final_params = initial_params
    return losses, final_params



def evaluation_metrics(original, pred):
    count = 0
    # print(len(original)==len(pred))
    for i in range(len(original)):
        if(original[i]== pred[i]):
            count+=1
    accuracy = count/len(original)

    # extract the different classes
    classes = np.unique(original)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((pred == classes[i]) & (original== classes[j]))
    
    precision = 0
    f1_score = 0
    recall = 0
    cm = confmat
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    
    
    for i in range(len(classes)):
        if true_pos[i] != 0:
            precision += true_pos[i] / (true_pos[i] + false_pos[i])
    precision = precision / len(classes)
   
    for i in range(len(classes)):
        if true_pos[i] != 0:
            recall += true_pos[i] / (true_pos[i] + false_neg[i])
    recall = recall / len(classes)


    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2*(precision*recall)/(precision + recall)

    return accuracy, precision, recall, f1_score

def main():
    #####################
    print('Correctness Verification 1')
    Train_X = np.asarray([[0.13000], [0.42000]])
    Train_Y = np.asarray([[0.90000], [0.23000]])
    
    init_params0 = initialize_network([1, 2, 1], scale=0.1) # initializes a sigle layer network (perceptron)
    init_params0 = {'W0': np.asarray([[0.1] ,[ 0.20000]]).T, 'b0': np.array([0.40000 , 0.30000]), 'W1': np.asarray([[ 0.5], [0.6]]), 'b1': np.asarray([0.7])}
    print('First Instance')
    prediction, cache = forward(init_params0, Train_X[0][None,:])
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y[0][None,:],alpha=0.25)
    print('Second Instance')
    prediction, cache = forward(init_params0, Train_X[1][None,:])
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y[1][None,:],alpha=0.25)

    print('Final')
    prediction, cache = forward(init_params0, Train_X)
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y,alpha=0.25)

    print("prediction: ", prediction)
    print("loss: ", loss)
    print("gradients: ", gradients)
    print()

    #####################
    print('Correctness Verification 2 2')
    Train_X = np.asarray([[0.32000, 0.68000],[0.83000, 0.02000]])
    Train_Y = np.asarray([[0.75000, 0.98000], [0.75000, 0.28000]])

    init_params0 = initialize_network([2, 4, 3,2], scale=0.1) # initializes a sigle layer network (perceptron)    
    init_params0 = {'W0': np.asarray(
    [[0.15000, 0.40000],
	[0.10000, 0.54000  ],
	[0.19000, 0.42000  ],
	[0.35000, 0.68000  ]]).T, 'b0': np.array([0.42000 , 0.72000,0.01,0.30]), 'W1': np.asarray([[0.67000, 0.14000,  0.96000,  0.87000],
	[0.42000,  0.20000,  0.32000,  0.89000],
	[0.56000,  0.80000,  0.69000,  0.09000 ]]).T, 'b1': np.asarray([0.21,0.87,0.03]),'W2': np.asarray([[0.87000,  0.42000,  0.53000],
	  [0.10000,  0.95000,  0.69000]]).T, 'b2': np.asarray([0.04,0.17])}
   

    print('First Instance')
    prediction, cache = forward(init_params0, Train_X[0][None,:])
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y[0][None,:],alpha=0.25)

    print('Second Instance')
    prediction, cache = forward(init_params0, Train_X[1][None,:])
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y[1][None,:],alpha=0.25)
    # losses0, final_params0 = gradient_descent(Train_X, Train_Y, init_params0, lr=1e-6, num_iterations=100)  
    print('Final')
    prediction, cache = forward(init_params0, Train_X)
    gradients, loss = backprop_and_loss(init_params0, prediction, cache, Train_Y,alpha=0.25)

    print("prediction: ", prediction)
    print("loss: ", loss)
    print("gradients: ", gradients)
    print()



    #########################
    print('Experiment 3')
    print('Voting.csv')
    voting_acc = []
    ttlist = create_k_folds("datasets/votes.csv", 10)
    accuracies = []
    f1_scores = []
    for tt in ttlist:
        Train_X = tt[0][:, :-1]
        Train_Y = tt[0][:, -1]
        Test_X = tt[1][:, :-1]
        Test_Y = tt[1][:, -1]
    
        N = Train_Y.shape[0]

        unique_labels = np.unique(Train_Y)
        n_labels = unique_labels.shape[0]
        Train_Y_processed = np.zeros((N,n_labels))
        Train_Y_processed[np.arange(N),Train_Y.astype(int)] = 1
        
        first_layer_size = Train_X.shape[1]
        output_layer_size = Train_Y_processed.shape[1]
    
        init_params0 = initialize_network([first_layer_size, 20, 20, output_layer_size], scale=1) # initializes a sigle layer network (perceptron)
        losses,final_params = gradient_descent(Train_X,Train_Y_processed,init_params0,lr=5e-3,num_iterations=10000, alpha=0)
        prediction, cache = forward(final_params, Test_X)
        y_prediction = np.argmax(prediction,axis=1)
        acc = np.sum(y_prediction == Test_Y)/Test_Y.shape[0]
        # voting_acc.append(acc)
        
        accuracy, precision, recall, f1_score = evaluation_metrics(list(y_prediction), list(Test_Y))
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
    print(sum(accuracies)/len(accuracies))
    print(sum(f1_scores)/len(f1_scores))
    print()
    # plt.plot(losses)
    # plt.show()

    ###########################
    

 
    print('Wine.csv')
   
    f1_scores = []
    wine_acc = []
    ttlist = create_k_folds("datasets/wine.csv", 10)
    for tt in ttlist:
        Train_X = tt[0][:, :-1]
        Train_Y = tt[0][:, -1]
        Test_X = tt[1][:, :-1]
        Test_Y = tt[1][:, -1]

        Train_X = (Train_X - np.mean(Train_X,axis=0)[None,:])/ np.std(Train_X,axis=0)[None,:] #normalization 
        Test_X = (Test_X - np.mean(Test_X,axis=0)[None,:])/np.std(Test_X,axis=0)[None,:] #normalization

        N = Train_Y.shape[0]

        unique_labels = np.unique(Train_Y)
        n_labels = unique_labels.shape[0]
        Train_Y_processed = np.zeros((N,n_labels))
        Train_Y_processed[np.arange(N),(Train_Y.astype(int) - 1)] = 1

        first_layer_size = Train_X.shape[1]
        output_layer_size = Train_Y_processed.shape[1]

        init_params0 = initialize_network([first_layer_size, 20, 20, output_layer_size], scale=1) # initializes a sigle layer network (perceptron)
        losses,final_params = gradient_descent(Train_X,Train_Y_processed,init_params0,lr=5e-3,num_iterations=10000, alpha = 0)
        prediction, cache = forward(final_params, Test_X)
        y_prediction = np.argmax(prediction,axis=1)
        acc = np.sum(y_prediction == (Test_Y.astype(int) - 1))/Test_Y.shape[0]
        wine_acc.append(acc)
        f1_scores.append(f1_score)
    print(sum(wine_acc)/len(wine_acc))
    print(sum(f1_scores)/len(f1_scores))
    print()


    plt.plot(losses)
    plt.show()
    ttlist = create_k_folds("datasets/votes.csv", 10)
    Train_X = ttlist[0][0][:, :-1]
    Train_Y = ttlist[0][0][:, -1]
    Test_X = ttlist[0][1][:, :-1]
    Test_Y = ttlist[0][1][:, -1]
    N = Train_Y.shape[0]
    unique_labels = np.unique(Train_Y)
    n_labels = unique_labels.shape[0]
    Train_Y_processed = np.zeros((N,n_labels))
    Train_Y_processed[np.arange(N),(Train_Y.astype(int) - 1)] = 1
    first_layer_size = Train_X.shape[1]
    output_layer_size = Train_Y_processed.shape[1]
    init_params0 = initialize_network([first_layer_size, 20, 20, output_layer_size], scale=1) # initializes a sigle layer network (perceptron)
    losses_f = []
    for i in range(1,len(Train_X)):
        train_X = Train_X[:i]
        train_y = Train_Y_processed[:i]
        prediction, cache = forward(init_params0, train_X)
        losses,final_params = gradient_descent(train_X,train_y,init_params0,lr=5e-2,num_iterations=100)
        losses_f.append(min(losses))
    plt.plot(losses_f)
    plt.show()
    
    ttlist = create_k_folds("datasets/wine.csv", 2)
    Train_X = ttlist[0][0][:, :-1]
    Train_Y = ttlist[0][0][:, -1]
    Test_X = ttlist[0][1][:, :-1]
    Test_Y = ttlist[0][1][:, -1]
    losses_f = []
    N = Train_Y.shape[0]

    N2 = Test_Y.shape[0]
    
    unique_labels = np.unique(Train_Y)
    n_labels = unique_labels.shape[0]
    Train_Y_processed = np.zeros((N,n_labels))
    Train_Y_processed[np.arange(N),(Train_Y.astype(int) - 1)] = 1

    Test_Y_processed = np.zeros((N2,n_labels))
    Test_Y_processed[np.arange(N2),(Test_Y.astype(int) - 1)] = 1


    first_layer_size = Train_X.shape[1]
    output_layer_size = Train_Y_processed.shape[1]
    init_params0 = initialize_network([first_layer_size, 20, 20, output_layer_size], scale=1) # initializes a sigle layer network (perceptron)
    for i in range(1,len(Train_X)):
        train_X = Train_X[:i]
        train_y = Train_Y_processed[:i]
        prediction, cache = forward(init_params0, train_X)
        losses,final_params = gradient_descent(train_X,train_y,init_params0,lr=5e0,num_iterations=100)
        prediction, cache = forward(init_params0, Test_X)
        gradients, loss = backprop_and_loss(init_params0, prediction, cache, Test_Y_processed)
        losses_f.append(loss)
    plt.plot(losses_f)
    plt.show()

    Train_X = np.asarray([[1,2,2,1],[2,1,5,2],[5,2,3,1]])
    Train_Y = np.asarray([[1],[0],[1]])



if __name__ == "__main__":
    main()


