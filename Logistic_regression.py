import re
import os,sys 
import numpy as np
import matplotlib.pyplot as plt

class Logistic_classifier(object):
    epsilon = 1e-16
    
    
    # Converts dictionary from dat file to a list
    def convert_dat_to_list(self,path):
        f = open(path,"r")
        if f.mode == 'r':
            contents = f.read()
            dict_list = contents.split('\n')
            return dict_list[:len(dict_list)-1]

    # Generate the feature vectors for training emails
    def generate_features(self,path_to_file,dict_list):
        feature_vector = [0] * len(dict_list)
        f = open(path_to_file,"r")
        if f.mode == 'r':
            contents = f.read()
            features = re.split('[.?,\s]\s*',contents)
            for x in features:
                if x in dict_list:
                    feature_vector[dict_list.index(x)]+=1
        return feature_vector
                    
    # Calculates the sigmoid function 
    def sigmoid(self,w,x,epsilon):
        x = np.matmul(x,w)
        sig = 1/(1 + np.exp(-x))
        sig = np.clip(sig,a_min = epsilon, a_max = 1 - epsilon)
        return sig
    
    # Calculates cross-entropy loss
    def cross_entropy(self,w,x,y,lambda_val):
        result = -np.mean(y*np.log(self.sigmoid(w,x,epsilon)) + (1-y)*np.log(1-self.sigmoid(w,x,epsilon))) + lambda_val * np.linalg.norm(w[1:],ord=2)/2
        return result

    # Calculates gradient of weights
    def gradient(self,w,x,y,lambda_val):
        grad = np.matmul(np.transpose(x),np.subtract(self.sigmoid(w,x,epsilon),y))/w.shape[0]
        reg = (2*lambda_val * w)/w.shape[0]
        reg[0] = 0
        return grad + reg

    # Run batch gradient descent for 50 iterations ( Change this value as needed )
    def gradient_descent(self,w,x,y,alpha,lambda_val):
        ce = []
        for i in range(50):
            ce.append(self.cross_entropy(w,x,y,lambda_val))
            grad_w = self.gradient(w,x,y,lambda_val)
            w = np.subtract(w,alpha * grad_w)
            
        l2_norm = np.linalg.norm(w[1:],ord=2)
        return ce,l2_norm

#Please uncomment the section below and give path as needed to the dictionary, spam emails, and ham emails
# path = '/Users/swetaharikumar/Desktop/Logistic_Regression/spam/dic.dat'
# path_to_spam = '/Users/swetaharikumar/Desktop/Logistic_Regression/spam/train/spam/'
# path_to_ham = '/Users/swetaharikumar/Desktop/Logistic_Regression/spam/train/ham/'

model = Logistic_classifier()
dict_list = model.convert_dat_to_list(path)
directory_spam = os.listdir(path_to_spam)
directory_ham= os.listdir(path_to_ham)

frequency_vector = [0] * len(dict_list)
feature_matrix = []
y = []

# Generate feature vectors as a matrix for spam
for files in directory_spam:
    feature_vector = model.generate_features(path_to_spam + files,dict_list)
    feature_matrix.append(feature_vector)
    frequency_vector = [frequency_vector[i] + feature_vector[i] for i in range(len(feature_vector))]
    y += [1]
    
# Append feature vectors from ham to the feature matrix
for files in directory_ham:
    feature_vector = model.generate_features(path_to_ham + files,dict_list)
    feature_matrix.append(feature_vector)
    frequency_vector = [frequency_vector[i] + feature_vector[i] for i in range(len(feature_vector))]
    y += [0]
    
# Adds the bias term to feature matrix
for row,fv in enumerate(feature_matrix):
    fv = [1] + fv
    feature_matrix[row] = fv


y_train = np.asarray(y).reshape(-1,1)
X_train = np.asarray(feature_matrix)

#Initilize w to zeros
w = np.zeros((np.shape(feature_matrix)[1],1))
w[0] = 0.1
np.shape(w)

#Alpha refers to step size
alpha=[0.001,0.01,0.05,0.1,0.5]

#Lambda_param refers to regularization parameter
lambda_param = [0,0.1,0.2,0.3,0.4,0.5]
ce_arr = []
l2_arr = []
for lr in alpha:
    ce,l2_norm = model.gradient_descent(w,X_train,y_train,lr,lambda_param[0])
    ce_arr.append(ce)
    l2_arr.append(l2_norm)


# Execute below code to see how cross entropy changes with respect to
# number of iterations for each step size chosen, without regularization
for i in range(len(alpha)):
    plt.plot(x_axis,ce_arr[i],label = 'alpha = '+ str(alpha[i]))
plt.title('Cross entropy values vs no of iterations for logistic regression without regularization')
plt.legend(loc = 'upper right')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy')
plt.show()

for i in range(len(alpha)):
    print("L2 norm at step size = "+ str(alpha[i]) + " is "+str(l2_arr[i]))

w = np.zeros((np.shape(feature_matrix)[1],1))
w[0] = 0.1
np.shape(w)

ce_arr = []
l2_arr = []
for lr in alpha:
    ce,l2_norm = model.gradient_descent(w,X_train,y_train,lr,lambda_param[1])
    ce_arr.append(ce)

#Execute below code to see how cross entropy varies with respect to
# number of iterations for each value of regularization parameter chosen
for i in range(len(alpha)):
    plt.plot(x_axis,ce_arr[i],label = 'lambda = '+ str(alpha[i]))
plt.title('Cross entropy values vs no of iterations for logistic regression with regularization')
plt.legend(loc = 'upper right')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy')
plt.show()

lambdas = [0,0.1,0.2,0.3,0.4,0.5]
l2_arr = []
for lam in lambdas:
    ce,l2_norm = model.gradient_descent(w,X_train,y_train,alpha[1],lam)
    l2_arr.append(l2_norm)

for i in range(len(lambdas)):
    print("L2 norm at reg parameter = "+ str(lambdas[i]) + " is "+str(l2_arr[i]))
