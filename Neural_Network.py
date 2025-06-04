import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryNeuralNetwork:
    """Single layer neural network for binary classification."""
    
    def __init__(self, input_dim: int, learning_rate: float = 0.1, 
                 init_method: str = "xavier"):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of input features
            learning_rate: Learning rate for gradient descent
            init_method: Weight initialization method ("xavier", "he", "standard")
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.weights = None
        self.bias = None
        self._initialize_parameters()
        
    def _initialize_parameters(self) -> None:
        """Initialize network parameters based on the chosen method."""
        if self.init_method == "xavier":
            # Xavier/Glorot initialization
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
        """Apply sigmoid activation function."""
        # Clip values to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss with regularization."""
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for weights and bias."""
        m = X.shape[0]
        error = y_pred - y_true
        dW = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        return dW, db
    
    def update_parameters(self, dW: np.ndarray, db: np.ndarray) -> None:
        """Update network parameters using gradient descent."""
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform one training step and return the loss."""
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass
        dW, db = self.compute_gradients(X, y, y_pred)
        
        # Update parameters
        self.update_parameters(dW, db)
        
        return loss
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        return (self.forward(X) >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model and return loss and accuracy."""
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        accuracy = np.mean((y_pred >= 0.5).astype(int) == y)
        return loss, accuracy

class ModelTrainer:
    """Handles the training process and visualization."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.results = []
        self.best_model = None
        self.best_accuracy = 0
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    learning_rates: List[float], init_methods: List[str],
                    epochs: int = 1000) -> None:
        """Train multiple models with different configurations."""
        for init_method in init_methods:
            for lr in learning_rates:
                logger.info(f"Training model with {init_method} initialization and learning rate {lr}")
                
                # Initialize model
                model = BinaryNeuralNetwork(self.input_dim, lr, init_method)
                
                # Training history
                train_losses = []
                test_losses = []
                
                # Training loop
                for epoch in range(epochs):
                    # Train step
                    train_loss = model.train_step(X_train, y_train)
                    
                    # Evaluate
                    test_loss, _ = model.evaluate(X_test, y_test)
                    
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                
                # Final evaluation
                _, train_accuracy = model.evaluate(X_train, y_train)
                _, test_accuracy = model.evaluate(X_test, y_test)
                
                # Store results
                self.results.append({
                    'init_method': init_method,
                    'learning_rate': lr,
                    'model': model,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy
                })
                
                # Update best model
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = test_accuracy
                    self.best_model = model
                
                logger.info(f"Training completed. Test accuracy: {test_accuracy:.4f}")
    
    def plot_learning_curves(self) -> None:
        """Generate and save learning curve plots."""
        # Color schemes for different parameters
        color_schemes = {
            'learning_rates': {0.01: 'navy', 0.1: 'darkgreen', 0.5: 'darkred'},
            'init_methods': {'xavier': 'purple', 'he': 'orange', 'standard': 'teal'}
        }
        
        # 1. Training Loss Comparison
        plt.figure(figsize=(12, 6))
        for result in self.results:
            color = color_schemes['learning_rates'][result['learning_rate']]
            plt.plot(result['train_losses'], 
                    label=f"LR: {result['learning_rate']}, Init: {result['init_method']}", 
                    color=color, alpha=0.7)
        
        plt.title("Training Loss Comparison", fontsize=12, pad=20)
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Binary Cross-Entropy Loss", fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Test Loss Comparison
        plt.figure(figsize=(12, 6))
        for result in self.results:
            color = color_schemes['init_methods'][result['init_method']]
            plt.plot(result['test_losses'], 
                    label=f"Init: {result['init_method']}, LR: {result['learning_rate']}", 
                    color=color, alpha=0.7)
        
        plt.title("Test Loss Comparison", fontsize=12, pad=20)
        plt.xlabel("Epochs", fontsize=10)
        plt.ylabel("Binary Cross-Entropy Loss", fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Grid of Learning Curves
        n_init_methods = len(set(r['init_method'] for r in self.results))
        n_learning_rates = len(set(r['learning_rate'] for r in self.results))
        
        fig, axes = plt.subplots(n_init_methods, n_learning_rates, 
                                figsize=(15, 10))
        fig.suptitle("Learning Curves Grid", fontsize=16, y=1.02)
        
        # Get loss limits for consistent scaling
        all_losses = []
        for result in self.results:
            all_losses.extend(result['train_losses'])
            all_losses.extend(result['test_losses'])
        
        y_min, y_max = min(all_losses), max(all_losses)
        y_margin = (y_max - y_min) * 0.1
        
        for i, init_method in enumerate(['xavier', 'he', 'standard']):
            for j, lr in enumerate([0.01, 0.1, 0.5]):
                ax = axes[i, j]
                
                # Find matching result
                result = next(r for r in self.results 
                            if r['init_method'] == init_method 
                            and r['learning_rate'] == lr)
                
                # Plot both train and test losses
                ax.plot(result['train_losses'], label='Train', 
                       color='blue', alpha=0.7)
                ax.plot(result['test_losses'], label='Test', 
                       color='red', alpha=0.7)
                
                ax.set_title(f"{init_method}, LR={lr}", fontsize=10)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.grid(True, alpha=0.3)
                
                if i == n_init_methods-1:
                    ax.set_xlabel("Epochs")
                if j == 0:
                    ax.set_ylabel("Loss")
                if i == 0 and j == n_learning_rates-1:
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig("learning_curves_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Model Performance Summary
        logger.info("\nModel Performance Summary:")
        for result in self.results:
            logger.info(
                f"\nInitialization: {result['init_method']}, "
                f"Learning Rate: {result['learning_rate']}"
            )
            logger.info(
                f"Train Accuracy: {result['train_accuracy']:.4f}, "
                f"Test Accuracy: {result['test_accuracy']:.4f}"
            )
            logger.info(
                f"Final Train Loss: {result['train_losses'][-1]:.4f}, "
                f"Final Test Loss: {result['test_losses'][-1]:.4f}"
            )
        
        # Best model information
        best_result = max(self.results, key=lambda x: x['test_accuracy'])
        logger.info("\nBest Model Performance:")
        logger.info(
            f"Initialization: {best_result['init_method']}, "
            f"Learning Rate: {best_result['learning_rate']}"
        )
        logger.info(f"Best Test Accuracy: {best_result['test_accuracy']:.4f}")
        
def main():
    # Load and preprocess data
    logger.info("Loading breast cancer dataset...")
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
                        learning_rates, init_methods)
    
    # Generate visualizations
    trainer.plot_learning_curves()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()