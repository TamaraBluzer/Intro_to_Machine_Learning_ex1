# 315287441 tamara bluzer
#211490362 itamar kolodny

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Union

class BinaryNeuralNetwork:
    
    def __init__(self, input_dim: int, learning_rate: float = 0.1, 
                 init_method: str = "xavier"):

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.weights = None
        self.bias = None
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        # initialize the weights and bias with3 different options
        if self.init_method == "xavier":
            # Xavier initialization
            limit = np.sqrt(6 / (self.input_dim + 1))
            self.weights = np.random.uniform(-limit, limit, (self.input_dim, 1))
        elif self.init_method == "he":
            # He initialization
            std = np.sqrt(2 / self.input_dim)
            self.weights = np.random.normal(0, std, (self.input_dim, 1))
        else:  # standard
            self.weights = np.random.normal(0, 0.1, (self.input_dim, 1))
            
        self.bias = np.zeros((1, 1))
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500) # we use the sigmoid activation func and clipping the input values to prevent numerical overflow
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        z = np.clip(z, -500, 500)
        return self.sigmoid(z) # applying the sigmoid activation function on X*w+b
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15 # prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) #cross entropy loss
    
    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = X.shape[0]
        error = y_pred - y_true
        dW = (1/m) * np.dot(X.T, error) # gradient for weights
        db = (1/m) * np.sum(error)      # gradient for bias
        return dW, db
    
    def update_parameters(self, dW: np.ndarray, db: np.ndarray) -> None:
        self.weights -= self.learning_rate * dW # updating weights
        self.bias -= self.learning_rate * db    # updating bias
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float: #single training step
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        dW, db = self.compute_gradients(X, y, y_pred)
        self.update_parameters(dW, db)
        return loss
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        # compute and return accuracy and loss
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        accuracy = np.mean((y_pred >= 0.5).astype(int) == y)
        return loss, accuracy

class ModelTrainer:
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.results = []
        self.best_model = None
        self.best_accuracy = 0
        self.X_train = None
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    learning_rates: List[float], init_methods: List[str],
                    epochs: int = 1000, convergence_threshold: float = 1e-4) -> None:

        # for each learning rate that we have and each initialization we will calculate the train and test loss and accuracy
        # this way we will find out what's the best initialization and learning rate
        self.X_train = X_train
        for init_method in init_methods:
            for lr in learning_rates:
                print(f"Training model with {init_method} initialization and learning rate {lr}")
                model = BinaryNeuralNetwork(self.input_dim, lr, init_method)
                train_losses = []
                test_losses = []
                
                for epoch in range(epochs):
                    train_loss = model.train_step(X_train, y_train)
                    test_loss, test_accuracy = model.evaluate(X_test, y_test)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    
                    # Check convergence
                    if epoch > 0 and convergence_threshold:
                        loss_change = abs(train_losses[-1] - train_losses[-2])
                        if loss_change < convergence_threshold:
                            print(f"Converged at epoch {epoch} (loss change: {loss_change:.6f})")
                            break
                
                _, train_accuracy = model.evaluate(X_train, y_train)
                _, test_accuracy = model.evaluate(X_test, y_test)
                
                self.results.append({
                    'init_method': init_method,
                    'learning_rate': lr,
                    'model': model,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'final_weights': model.weights.copy(),
                    'final_bias': model.bias.copy(),
                    'epochs_trained': len(train_losses)
                })
                
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = test_accuracy
                    self.best_model = model
                
                print(f"Training completed. Test accuracy: {test_accuracy:.4f}")
    
    def plot_learning_curves(self) -> None:
        # Training Loss Plot
        plt.figure(figsize=(12, 6))
        for result in self.results:
            label = f"{result['init_method']}, lr={result['learning_rate']}"
            plt.plot(result['train_losses'], label=label, linewidth=2)
        
        plt.title("Training Loss Over Time", fontsize=14, pad=20)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Test Loss Plot
        plt.figure(figsize=(12, 6))
        for result in self.results:
            label = f"{result['init_method']}, lr={result['learning_rate']}"
            plt.plot(result['test_losses'], label=label, linewidth=2)
        
        plt.title("Test Loss Over Time", fontsize=14, pad=20)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test_loss.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Training configurations
    learning_rates = [0.01, 0.1, 0.5]
    init_methods = ["xavier", "he", "standard"]
    
    # Initialize and run trainer
    trainer = ModelTrainer(input_dim=X.shape[1])
    trainer.train_models(X_train, y_train, X_test, y_test,
                        learning_rates, init_methods,
                        epochs=1000,
                        convergence_threshold=1e-4)
    
    # Generate visualizations
    trainer.plot_learning_curves()
    
    print("\nModel Performance Summary:")
    for result in trainer.results:
        print(f"\nInitialization: {result['init_method']}, Learning Rate: {result['learning_rate']}")
        print(f"Train Accuracy: {result['train_accuracy']:.4f}, Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Final Loss: {result['train_losses'][-1]:.4f}")
        print(f"Epochs trained: {result['epochs_trained']}")
    
    print(f"\nBest Model:")
    best_result = max(trainer.results, key=lambda x: x['test_accuracy'])
    print(f"Initialization: {best_result['init_method']}, Learning Rate: {best_result['learning_rate']}")
    print(f"Best Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"\nFinal Weights Shape: {best_result['final_weights'].shape}")
    print(f"First few weights: {best_result['final_weights'][:5].flatten()}")
    print(f"Final Bias: {best_result['final_bias'].flatten()[0]:.6f}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()