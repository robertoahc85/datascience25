import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/digit/train.csv')
train.head()

unprocessed_y_train = train.label
unprocessed_y_train.head()

def vectorized_result(j):
    e= np.zeros(10)
    e[j] = 1.0
    return e

y_1 = vectorized_result(unprocessed_y_train[3])
print(y_1)


# In[17]:


y_train = pd.DataFrame([vectorized_result(y) for y in unprocessed_y_train])
y_train.head()


# In[18]:


x_train = train.drop('label',axis=1)
x_train.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)


# In[19]:


print(y_train.size)
print(x_train.size)
print(x_train.size/y_train.size)


# In[20]:


def sigmoid(t):
   return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
   return p * (1 - p)


# In[21]:


class NeuralNetwork:
    def __init__(self, x,y):
        self.lr = 0.2
        self.input = x
        self.weights1 = 2*np.random.rand(self.input.shape[1],16)-1 # considering we have 16 nodes in the hidden layer
        self.bias1 = 2 * np.random.rand(1, 16) - 1
        self.weights2 = 2*np.random.rand(16,16)-1
        self.bias2 = 2 * np.random.rand(1, 16) - 1
        self.weightsOutput = np.random.rand(16,10)
        self.biasOut = 2 * np.random.rand(1, 10) - 1
        self.y = y
        self.output = np.zeros(y.shape)
        self.errors = []
    
    def feedforward(self, data = None):
        #np.dot regresa el producto punto de dos arreglos.
        if(data is None):
            data = self.input
        self.layer1 = sigmoid(np.dot(data, self.weights1)+self.bias1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2)+self.bias2)
        self.output = sigmoid(np.dot(self.layer2, self.weightsOutput)+self.biasout)
        return self.output
        
    def backprop(self):
        # Aplicando la regla de la cadena para la función de perdida respecto a los pesos 2 y 1.
        # .T devuelve la matriz traspuesta
        mse = np.sum((self.y - self.output)**2)
        self.errors.append(mse)
        aux = 2*(self.y - self.output) * sigmoid_derivative(self.output)
        d_weightsOutput = np.dot(self.layer2.T, aux)
        d_weights2 = np.dot(self.layer1.T,  (np.dot(aux, self.weightsOutput.T) * sigmoid_derivative(self.layer2)))
        aux = (np.dot(aux, self.weightsOutput.T) * sigmoid_derivative(self.layer2))
        d_weights1 = np.dot(self.input.T,  (np.dot(aux, self.weights2.T) * sigmoid_derivative(self.layer1)))
                
        # Actualización de los pesos a través del uso de la pendiente (derivada) de la función de perdida.
        self.weights1 += d_weights1 * self.lr
        self.weights2 += d_weights2 *self.lr
        self.weightsOutput += d_weightsOutput *self.lr
        
    def fit(self):
        self.output = self.feedforward()
        self.backprop()
        
    predict = feedforward


# In[11]:


NN = NeuralNetwork(x_train,y_train)
for i in range(10): # trains the NN 1,000 times
    if i % 1 == 0:
        print ("for iteration # " + str(i) + "\n")
        #print ("Input : \n" + str(x_train))
        print ("Actual Output: \n" + str(y_train))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.mean(np.square(y_train - NN.feedforward()), axis=0)))) # mean sum squared loss
        print ("\n")
  
    NN.fit()

print()

