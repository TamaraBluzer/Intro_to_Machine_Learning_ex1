#noa zisbord 315241414
#nevo heller 322401662
#rotem vasa 322209529

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y_true = data.target.reshape(-1, 1)
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_true[:split_idx], y_true[split_idx:]

#standardize the features (subtract mean and divide by standard deviation)
#this step makes sure the learning process will not be effected
#by the nature of the information in the data set
#for example, the radius value can be much smaller then
#the perimeter value, which can effect the learning process
#as explained in https://stats.stackexchange.com/questions/458579/should-i-normalize-all-data-prior-feeding-the-neural-network-models
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_train = standardize(X_train)
X_test = standardize(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def initialize_weights(input_dim, output_dim, init_type="uniform"):
    if init_type == "uniform":
        weights = np.random.uniform(-1, 1, (input_dim, output_dim))
    elif init_type == "normal":
        weights = np.random.normal(0, 1, (input_dim, output_dim))
    elif init_type == "random":
        # Completely random initialization (values from -1 to 1)
        weights = np.random.random((input_dim, output_dim)) * 2 - 1  # Values between -1 and 1
    else:
        raise ValueError("Invalid initialization type")
    biases = np.zeros((1, output_dim))
    return weights, biases

def compute_gradients(X, y_true, y_pred):
    m = X.shape[0]
    error = y_pred - y_true
    dW = np.dot(X.T, error) / m
    db = np.sum(error, axis=0, keepdims=True) / m
# the math behind the db dW derivation is explained in the pdf.
    return dW, db

def gradient_descent(weights, biases, dW, db, learning_rate):
    weights -= learning_rate * dW
    biases -= learning_rate * db
    return weights, biases

def get_axis_limits(results):
    all_train_losses = [result[2] for result in results]
    all_test_losses = [result[3] for result in results]
    min_train_loss = min([min(loss) for loss in all_train_losses])
    max_train_loss = max([max(loss) for loss in all_train_losses])
    min_test_loss = min([min(loss) for loss in all_test_losses])
    max_test_loss = max([max(loss) for loss in all_test_losses])
    return (min_train_loss, max_train_loss), (min_test_loss, max_test_loss)

learning_rates = [0.01, 0.1, 0.5]
initializations = ["uniform", "normal", "random"]
results = []


for init_type in initializations:
    for lr in learning_rates:
        weights, biases = initialize_weights(X_train.shape[1], 1, init_type=init_type)
        train_losses = []
        test_losses = []
        for epoch in range(1000):
            y_pred_train = sigmoid(np.dot(X_train, weights) + biases)
            train_loss = cross_entropy_loss(y_train, y_pred_train)
            train_losses.append(train_loss)

            y_pred_test = sigmoid(np.dot(X_test, weights) + biases)
            test_loss = cross_entropy_loss(y_test, y_pred_test)
            test_losses.append(test_loss)

            dW, db = compute_gradients(X_train, y_train, y_pred_train)
            weights, biases = gradient_descent(weights, biases, dW, db, lr)
        results.append((init_type, lr, train_losses, test_losses, weights, biases))

#color dict. to use for the plotting
color_map_lr = {0.01: 'dodgerblue', 0.1: 'orange', 0.5: 'green'}
color_map_init = {"uniform": 'teal', "normal": 'gold',"random":'crimson' }
# Plotting loss curves for learning rates
plt.figure(figsize=(12, 6))
for init_type in initializations:
    for lr in learning_rates:
        for result in results:
            if result[0] == init_type and result[1] == lr:
                color = color_map_lr[lr]  # Use color map by learning rate
                plt.plot(result[2], label=f"Train Loss - LR: {lr}", color=color)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_learning_rates.png")
plt.close()

# Plotting loss curves for initializations
plt.figure(figsize=(12, 6))
for lr in learning_rates:
    for init_type in initializations:
        for result in results:
            if result[0] == init_type and result[1] == lr:
                color = color_map_init[init_type]
                plt.plot(result[3], label=f"Test Loss - Init: {init_type}, LR: {lr}", color=color)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss for Different Initializations")
plt.legend()
plt.grid(True)
plt.savefig("test_loss_initializations.png")
plt.close()

# Plotting a table of graphs
fig, axs = plt.subplots(len(initializations), len(learning_rates), figsize=(15, 10))
fig.suptitle("Loss Curves for Different Learning Rates and Initializations", fontsize=16)

for i, init_type in enumerate(initializations):
    for j, lr in enumerate(learning_rates):
        result = results[i * len(learning_rates) + j]
        train_losses, test_losses = result[2], result[3]
        ax = axs[i, j]
        ax.plot(train_losses, label=f"Train")
        ax.plot(test_losses, label=f"Test")
        ax.set_title(f"Init: {init_type}, LR: {lr}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("loss_curves_table.png")
plt.close()

#normlized grid graphs -
train_loss_limits, test_loss_limits = get_axis_limits(results)
fig, axs = plt.subplots(len(initializations), len(learning_rates), figsize=(15, 10))
fig.suptitle("Loss Curves for Different Learning Rates and Initializations: fixed grid", fontsize=16)

for i, init_type in enumerate(initializations):
    for j, lr in enumerate(learning_rates):
        result = results[i * len(learning_rates) + j]
        train_losses, test_losses = result[2], result[3]
        ax = axs[i, j]
        ax.plot(train_losses, label=f"Train")
        ax.plot(test_losses, label=f"Test")
        ax.set_title(f"Init: {init_type}, LR: {lr}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(train_loss_limits)
        ax.set_xlim(0, 1000)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("loss_curves_table_norm.png")
plt.close()

#printing parameters for analyzing
best_loss_model = None
best_accuracy_model = None
lowest_loss = float('inf')
highest_accuracy = 0

#accuracy calc +determine the best models.
for init_type, lr, train_losses, test_losses, weights, biases in results:
    final_train_pred = sigmoid(np.dot(X_train, weights) + biases) >= 0.5
    final_test_pred = sigmoid(np.dot(X_test, weights) + biases) >= 0.5
    train_accuracy = np.mean(final_train_pred == y_train)
    test_accuracy = np.mean(final_test_pred == y_test)

    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]

    if final_test_loss < lowest_loss:
        lowest_loss = final_test_loss
        best_loss_model = (init_type, lr, weights, biases, final_train_loss, final_test_loss)

    if test_accuracy > highest_accuracy:
        highest_accuracy = test_accuracy
        best_accuracy_model = (init_type, lr, weights, biases, train_accuracy, test_accuracy)

    print(f"Init Type: {init_type}, Learning Rate: {lr}")
    print(f"Final Train Loss: {final_train_loss:.4f}, Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Train Accuracy: {train_accuracy:.4f}, Final Test Accuracy: {test_accuracy:.4f}\n")

#best loss model
print("Model with the Lowest Final Test Loss:")
print(f"Init Type: {best_loss_model[0]}, Learning Rate: {best_loss_model[1]}")
print(f"Final Train Loss: {best_loss_model[4]:.4f}, Final Test Loss: {best_loss_model[5]:.4f}")
print(f"Biases: {best_loss_model[3]}")
print(f"Weights: {best_loss_model[2].T}\n")

#best accuracy model
print("Model with the Highest Test Accuracy:")
print(f"Init Type: {best_accuracy_model[0]}, Learning Rate: {best_accuracy_model[1]}")
print(f"Final Train Accuracy: {best_accuracy_model[4]:.4f}, Final Test Accuracy: {best_accuracy_model[5]:.4f}")
print(f"Biases: {best_accuracy_model[3]}")
print(f"Weights: {best_accuracy_model[2].T}\n")