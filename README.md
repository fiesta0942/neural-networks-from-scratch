# Neural Networks from Scratch

Based on Deep Learning Specialization, deeplearning.ai, this is my trial on implementation of logistic regression, shallow / deep neural networks from scratch using NumPy

# 1. Logistic Regression
- Based on Breast Cancer Dataset from scikit-learn, I've implemented forward / backward propagation using Sigmoid as activation. 

# 2. Shallow Neural Network
- Based on make_moons dataset, which requires nonlinear decision boundary, I've implemented Neural Network with one hidden layer.
- It uses tanh function as activation in hidden layer and sigmoid function in the output layer.
- After training, it showed 97.12% accuracy for training set and 98.00% for test set.


# 3. Deep Neural Network
- Based on CIFAR-10 image dataset, I've implemented Deep Neural Network with 2 hidden layer. 
- It used ReLU activation function in hidden layer and softmax function to classify 10 classes.
- I've implemented two backward propagation: linear and ReLU and understood how backward propagation work for multi hidden layer, along with how caches work.