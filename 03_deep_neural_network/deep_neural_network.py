# Deep Neural Network with L - Hidden Layers

# Code by Minsoo Kang (Kyung Hee University, Department of Mathematics)

import numpy as np
from keras.datasets import cifar10

# 1. Load CIFAR10 Datasets

# 1.1 Original X = (50000, 32, 32, 3) , Y = (50000, 1)
(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = cifar10.load_data()

# 1.2 Flatten X into 1D Matrix
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T     # (3072, 50000) , 3072 feature vectors with 50000 samples
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T        # (3072, 10000)

# 1.3 Normalize X: 0 ~ 255 -> 0 ~ 1
X_train = X_train_flatten / 255.0
X_test = X_test_flatten / 255.0

# 1.4 One-hot code Y -> Make it 10 classification 
Y_train = np.eye(10)[Y_train_orig.flatten()].T                           # (10, 50000) , 10개의 classes(dog,cat,plane,...) 50000개의 sample
Y_test = np.eye(10)[Y_test_orig.flatten()].T                            # (10, 10000)


###################################################################################################################

# 2. Basic Funtions

def initialize_parameters(layer_dims):
    # 나중에 함수를 부를 때 layer_dims = [...]식으로 Layer 개수를 정할 수 있다
    # 만약 [3072, 50, 20, 10] 이라면 3072개의 input feature, 첫번째 hidden layer에 50개의 노드, 2번째에 20개의 노드, output엔 10개의 노드
    # [3072, 50, 20, 10]은 "Input Layer까지 포함해서" 4개의 layer. 실제 L은 -1을 해줘야 함
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims) - 1
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])    # Dimension of W = (현재 layer의 node 개수, 직전 layer의 node 개수
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))                              # Dimension of b = (현재 layer의 node 개수, 1)
        
    return parameters       # Create empty dictionary 'parameters' and label W{l}, b{l}


def forward_propagation(A, W, b):
    Z = np.dot(W,A) + b
    linear_cache = (A, W, b)
    
    return Z, linear_cache

def relu(Z):
    A = np.maximum(0,Z)     # ReLu
    activation_cache = Z    # Cache for backward propagation
    
    return A, activation_cache

def softmax(Z):             # Since there are 10 outputs, sigmoid(0 or 1) cannot be used
    shift_Z = Z - np.max(Z, axis=0, keepdims = True)
    exps = np.exp(shift_Z)
    
    A = exps / np.sum(exps, axis = 0 ,keepdims=True)
    activation_cache = Z
    
    return A, activation_cache

def activation_forward_propagation(A_prev, W,b, activation):
    
    if activation == "relu":
        Z, linear_cache = forward_propagation(A_prev, W, b) # np.dot(A_prev, W) + b를 계산
        A, activation_cache = relu(Z)
        
    else:
        Z, linear_cache = forward_propagation(A_prev, W,b)
        A, activation_cache = softmax(Z)
        
    layer_cache = (linear_cache, activation_cache)
    
    return A, layer_cache
    
    
def compute_cost(AL, Y):         # Calculate the cost function
    # AL: Output after softmax (Final Estimation)
    # Y: one - hot label (Real Answer)
    
    m = Y.shape[1]
    cost = - np.sum (Y * np.log(AL + 1e-8)) / m     # Cross - Entropy Loss. Check mathematical progress!
    
    return cost

def backward_propagation(dZ, linear_cache):            # 원래는 dA를 구하고 dZ를 구해야함. 하지만 수학적으로 dZ = AL - Y가 나와버림. 즉 dA 구하는건 skip
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy = True)
    
    dZ[Z <= 0] = 0 
    
    return dZ

def activation_backward_propagation(dA, layer_cache, activation):
    linear_cache, activation_cache = layer_cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = backward_propagation(dZ, linear_cache)
        
    return dA_prev, dW, db      #dA_prev는 다음 층으로 전달 / dW,db는 updating parameter용


###################################################################################################################
# 3. L - Model

def L_model_forward (X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # [3072, 50, 20, 10]일 경우 Hidden Layer가 2개이므로 W,b도 2개씩 = 2
    
    for l in range (1,L):       # ReLu는 1층부터 L-1층까지
        A_prev = A
        A, layer_cache = activation_forward_propagation(A_prev,
                                                  parameters[f'W{l}'],
                                                  parameters[f'b{l}'],
                                                  activation = "relu")
        caches.append(layer_cache)
        
    AL, layer_cache = activation_forward_propagation(A,                       # L-1층 이후 나온 A로 softmax 계산
                                                parameters[f'W{L}'],
                                                parameters[f"b{L}"],
                                                activation = "softmax")
    caches.append(layer_cache)
        
    return AL, caches   # caches에는 X,W1,b1,Z1 / A1, W2, b2 ,Z2 / A2, W3, b3 ,Z3

def L_model_backward (AL, Y, caches):
    grads = {}
    L = len(caches) # 전체 층의 개수
    m = AL.shape[1]
    
    dZ = AL - Y
    
    layer_cache = caches[L-1]
    linear_cache, activation_cache = layer_cache
    dA_prev, dW, db = backward_propagation(dZ, linear_cache)
    
    grads[f'dA{L-1}'] = dA_prev
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db
    
    for l in reversed(range(1,L)):
        layer_cache = caches [l - 1]
        dA = grads[f'dA{l}']
        dA_prev, dW, db = activation_backward_propagation(dA, layer_cache, activation = "relu")
        
        grads[f'dA{l-1}'] = dA_prev
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
        
    return parameters
    
    
###################################################################################################################
     
# 4. Train

def L_layer_model(X,Y,layers_dims, learning_rate, num_iterations, print_cost = True):
    np.random.seed(42)
    costs = []
    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost (AL,Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:f}")
            costs.append(cost)
            
    return parameters, costs

###################################################################################################################

# 5 Run
layers_dims = [3072, 50, 20, 10]
trained_parameters, costs = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.005, num_iterations=1000, print_cost = True)

                                                
                                        