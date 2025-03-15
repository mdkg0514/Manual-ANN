import numpy as np
import pandas as pd
import math
def initialize_parameters(initialize_method, architecture = [6, 10, 1]):
    weights_matrix = {}
    bias_matrix = {}
    # architecture = architecture.insert(0, input_col)
    if initialize_method == 'Xavier Normal':
      # Xavier/Glorat Normal Initialization (Works well with activation functions like Sigmoid, Tanh)
      for i in range(1, len(architecture)):
          factor = math.sqrt(1 / architecture[i-1])
          weights_matrix[f'(l{i})'] = np.random.randn(architecture[i-1], architecture[i]) * factor
          bias_matrix[f'(l{i})'] = np.random.randn(1, architecture[i]) * factor
    elif initialize_method == 'Xavier Uniform':
      # Xavier/Glorat Uniform Initialization (Works well with activation functions like Sigmoid, Tanh)
      for i in range(1, len(architecture)):
          # It just calculates factor as math.sqrt(6 / (no_weights_in + no_weights_out))
          if i == 2:
              factor = math.sqrt(6 / (architecture[i-1] + architecture[i+1]))
          else:
              factor = math.sqrt(6 / (architecture[i-1]))
          weights_matrix[f'(l{i})'] = np.random.randn(architecture[i-1], architecture[i]) * factor
          bias_matrix[f'(l{i})'] = np.random.randn(1, architecture[i]) * factor
    elif initialize_method == 'He Normal':
      # He Normal Initialization (Works well with activation functions like Relu)
      for i in range(1, len(architecture)):
          factor = math.sqrt(2 / architecture[i])
          weights_matrix[f'(l{i})'] = np.random.randn(architecture[i-1], architecture[i]) * factor
          bias_matrix[f'(l{i})'] = np.random.randn(1, architecture[i]) * factor
    elif initialize_method == 'He Uniform':
      # He Uniform Initialization (Works well with activation functions like Relu)
      for i in range(1, len(architecture)):
          factor = math.sqrt(6 / architecture[i])
          weights_matrix[f'(l{i})'] = np.random.randn(architecture[i-1], architecture[i]) * factor
          bias_matrix[f'(l{i})'] = np.random.randn(1, architecture[i]) * factor
    return weights_matrix, bias_matrix

def EWMA(X, beta = 0.9):
    ewma = []
    ewma_refined = []
    for i in range(X.shape[0]):
        if i == 0:
            ewma.append(X[i])
            ewma_refined.append((X[i]).tolist())
        else:
            ewma.append(beta * ewma[i-1] + (1 - beta) * X[i])
            ewma_refined.append((beta * ewma[i-1] + (1 - beta) * X[i]).tolist())
    return np.array(ewma_refined)


def NAG(w, derivative_la, vt_1):
    eta = 0.01
    beta = 0.9
    # w_la = w - (beta * vt_1)
    v_t = (beta * vt_1) + (eta * derivative_la)
    w = w - v_t
    return w

def Momentum_SGD(w, derivative, vt_1):
    eta = 0.01
    beta = 0.9
    w = w - (eta * derivative) - (beta * vt_1)
    return w

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x):
    return max(x*0.01, x)

def parametric_relu(x, a):
    return max(x*a, x)

def elu(x, a):
    return max(a*(np.exp - 1), x)

def binary_cross_entropy_lf(true_val, pred_val):
    each_bce = -(((true_val * np.log(pred_val)) + ((1 - true_val) * np.log(1 - pred_val))))
    return each_bce

def binary_cross_entropy_cf(true_val, pred_val):
    each_bce = 0
    true_val = true_val.reshape(-1)
    for i in range(true_val.shape[0]):
        each_bce += ((true_val[i] * np.log(pred_val[i])) + ((1 - true_val[i]) * np.log(1 - pred_val[i])))
    return -each_bce/(true_val.shape[0])

def forward_propagation(X, y, weights, bias):
    output = {}
    for key, value in weights.items():
        w = value
        try:
            x = X.reshape(1, 6)
        except:
            x = X.reshape(1, X.shape[1])
        b = bias[key]
        pred = sigmoid((x @ w) + b)
        output[key] = pred
        X = pred
    # print(pred)
    loss = binary_cross_entropy_lf(y, pred)
    return output, loss

def L2_Regularization(w, derivative, lamda = 0.1):
    # lamda -----> If high value, then high impact regularization and vice vera
    eta = 0.01
    w = (1 - eta * lamda) * w - (eta * derivative)
    return w

def GD(w, derivative):
    eta = 0.01
    w = w - (eta * derivative)
    return w

def back_propagation(X, y, activations, weights, bias):
    for layer_no in range(2, 0, -1):
        # print(f"Layer {layer_no}:")
        dl_dy_hat = -(y - activations['(l2)'].reshape(-1)[0])
        activations['(l0)'] = X
        nipl = len(activations[f"(l{layer_no - 1})"].reshape(-1)) # neurons in previous layer
        nicl = len(activations[f"(l{layer_no})"].reshape(-1)) # neurons in current layer
        for neuron_c in range(nicl):
            # prev_args = [dl_dy_hat]
            for neurons_p in range(nipl):
                acti = activations[f"(l{layer_no - 1})"].reshape(-1)[neurons_p]
                if layer_no == 2:
                    wgt = 1
                else:
                    wgt = weights[f'(l{layer_no + 1})'].reshape(-1)[neuron_c]
                weight_derivative = dl_dy_hat * acti * wgt
                w_old = weights[f'(l{layer_no})'][neurons_p][neuron_c].reshape(-1)[0]
                w_new = GD(w_old, weight_derivative)
                weights[f'(l{layer_no})'][neurons_p][neuron_c] = w_new
            b_old = bias[f'(l{layer_no})'].reshape(-1)[neuron_c]
            bias_derivative = dl_dy_hat * wgt
            bias_new = GD(b_old, bias_derivative)
            bias[f'(l{layer_no})'][0][neuron_c] = bias_new
    return weights, bias

def back_propagation_MSGD(X, y, activations, weights, bias):
    for layer_no in range(2, 0, -1):
        # print(f"Layer {layer_no}:")
        dl_dy_hat = -(y - activations['(l2)'].reshape(-1)[0])
        activations['(l0)'] = X
        nipl = len(activations[f"(l{layer_no - 1})"].reshape(-1)) # neurons in previous layer
        nicl = len(activations[f"(l{layer_no})"].reshape(-1)) # neurons in current layer
        vts = EWMA(activations[f"(l{layer_no - 1})"].reshape(-1))
        for neuron_c in range(nicl):
            # prev_args = [dl_dy_hat]
            for neurons_p in range(nipl):
                acti = activations[f"(l{layer_no - 1})"].reshape(-1)[neurons_p]
                if layer_no == 2:
                    wgt = 1
                else:
                    wgt = weights[f'(l{layer_no + 1})'].reshape(-1)[neuron_c]
                weight_derivative = dl_dy_hat * acti * wgt
                w_old = weights[f'(l{layer_no})'][neurons_p][neuron_c].reshape(-1)[0]
                if neurons_p == 0:
                    w_new = Momentum_SGD(w_old, weight_derivative, vts[neurons_p])
                else:
                    w_new = Momentum_SGD(w_old, weight_derivative, vts[neurons_p -  1])
                weights[f'(l{layer_no})'][neurons_p][neuron_c] = w_new
            b_old = bias[f'(l{layer_no})'].reshape(-1)[neuron_c]
            bias_derivative = dl_dy_hat * wgt
            bias_new = Momentum_SGD(b_old, bias_derivative)
            bias[f'(l{layer_no})'][0][neuron_c] = bias_new
    return weights, bias

def back_propagation_NAG(X, y, activations, weights, bias):
    for layer_no in range(2, 0, -1):
        # print(f"Layer {layer_no}:")
        dl_dy_hat = -(y - activations['(l2)'].reshape(-1)[0])
        activations['(l0)'] = X
        nipl = len(activations[f"(l{layer_no - 1})"].reshape(-1)) # neurons in previous layer
        nicl = len(activations[f"(l{layer_no})"].reshape(-1)) # neurons in current layer
        vts = EWMA(activations[f"(l{layer_no - 1})"].reshape(-1))
        for neuron_c in range(nicl):
            # prev_args = [dl_dy_hat]
            for neurons_p in range(nipl):
                acti = activations[f"(l{layer_no - 1})"].reshape(-1)[neurons_p]
                if layer_no == 2:
                    wgt = 1
                else:
                    wgt = weights[f'(l{layer_no + 1})'].reshape(-1)[neuron_c]
                w_la = wgt - (beta * vt_1)
                weight_derivative = dl_dy_hat * acti * w_la
                w_old = weights[f'(l{layer_no})'][neurons_p][neuron_c].reshape(-1)[0]
                if neurons_p == 0:
                    w_new = NAG(w_old, weight_derivative, vts[neurons_p])
                else:
                    w_new = NAG(w_old, weight_derivative, vts[neurons_p -  1])
                weights[f'(l{layer_no})'][neurons_p][neuron_c] = w_new
            b_old = bias[f'(l{layer_no})'].reshape(-1)[neuron_c]
            bias_derivative = dl_dy_hat * w_la
            bias_new = NAG(b_old, bias_derivative)
            bias[f'(l{layer_no})'][0][neuron_c] = bias_new
    return weights, bias

def manual_model_training(X, y, architecture, epochs):
    weights, bias = initialize_parameters("Xavier Normal")
    print(weights)
    for epoch in range(epochs):
        print(f'Epoch no. {epoch+1}')
        pred = []
        for i in range(X.shape[0]):
            # print(f"\nInput {i+1}:")
            activations, loss = forward_propagation(X[i], y[i], weights, bias)
            weights, bias = back_propagation(X[i], y.reshape(-1)[i], activations, weights, bias)
            pred.append(activations['(l2)'].reshape(-1)[0])
        loss = binary_cross_entropy_cf(y, pred)
        print("Loss:", loss)
    # return pred, loss
    return loss

df = pd.read_csv("C:/Users/dwdqb/Desktop/Python Programs/SKLearn/Titanic.csv")
df.dropna(inplace = True)
y = df[['Survived']].values
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
X['Family'] = df['SibSp'] + df['Parch']
X = X.drop(['SibSp', 'Parch'], axis = 1)
mapped_values = {
    'C':0,
    'S':1,
    'Q':2,
    'female':1,
    'male':0
}
X['Sex'] = X['Sex'].map(mapped_values)
X['Embarked'] = X['Embarked'].map(mapped_values)
X = X.values
k = X
X = X / np.max(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

architecture = [6, 10, 1]
loss = manual_model_training(X_train, y_train, architecture, 100)
