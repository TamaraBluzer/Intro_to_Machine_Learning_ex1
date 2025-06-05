import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from typing import List, Union, Dict, Tuple

class TreeNode:
    """A class representing a node in the decision tree"""
    def __init__(self, feature_index=None, threshold=None, feature_type=None,
                 left_child=None, right_child=None, predicted_class=None):
        self.feature_index = feature_index      # Index of the feature to split on
        self.threshold = threshold              # Threshold value for numerical, category for categorical
        self.feature_type = feature_type        # 'numerical' or 'categorical'
        self.left_child = left_child           # Left subtree
        self.right_child = right_child         # Right subtree
        self.predicted_class = predicted_class  # Class prediction (for leaf nodes)

class DecisionTree:
    """Decision Tree Classifier using ID3 algorithm"""
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None
        self.feature_types = None
    
    def entropy(self, y):
        """Calculate entropy of a node"""
        # Get probability of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def information_gain(self, X, y, feature_idx, threshold, feature_type):
        """Calculate information gain for a split"""
        parent_entropy = self.entropy(y)
        
        # Create split based on feature type
        if feature_type == 'numerical':
            left_mask = X[:, feature_idx] <= threshold
        else:  # categorical
            left_mask = X[:, feature_idx] == threshold
            
        right_mask = ~left_mask
        
        # If split is empty, return 0 gain
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = n - n_left
        
        entropy_left = self.entropy(y[left_mask])
        entropy_right = self.entropy(y[right_mask])
        
        # Calculate information gain
        weighted_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
        information_gain = parent_entropy - weighted_entropy
        
        return information_gain
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold to split the data"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_type = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_type = self.feature_types[feature_idx]
            unique_values = np.unique(X[:, feature_idx])
            
            if feature_type == 'numerical':
                # For numerical features, try midpoints between values
                if len(unique_values) > 1:
                    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                else:
                    continue
            else:  # categorical
                # For categorical features, try each unique value
                thresholds = unique_values
            
            # Try all possible thresholds
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold, feature_type)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_type = feature_type
        
        return best_feature, best_threshold, best_gain, best_type
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < 2:
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(predicted_class=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain, best_type = self.find_best_split(X, y)
        
        # If no improvement, make a leaf node
        if best_gain <= 0:
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(predicted_class=leaf_value)
        
        # Create child nodes based on feature type
        if best_type == 'numerical':
            left_mask = X[:, best_feature] <= best_threshold
        else:  # categorical
            left_mask = X[:, best_feature] == best_threshold
            
        right_mask = ~left_mask
        
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(best_feature, best_threshold, best_type, left_tree, right_tree)
    
    def fit(self, X, y, feature_names=None, feature_types=None):
        """Train the decision tree"""
        self.feature_names = feature_names
        # If feature types not provided, assume all numerical
        self.feature_types = feature_types if feature_types else ['numerical'] * X.shape[1]
        self.root = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, node):
        """Predict single sample"""
        if node.predicted_class is not None:
            return node.predicted_class
        
        if node.feature_type == 'numerical':
            if x[node.feature_index] <= node.threshold:
                return self.predict_sample(x, node.left_child)
            return self.predict_sample(x, node.right_child)
        else:  # categorical
            if x[node.feature_index] == node.threshold:
                return self.predict_sample(x, node.left_child)
            return self.predict_sample(x, node.right_child)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(x, self.root) for x in X])
    
    def print_tree(self, node=None, indent="", feature_names=None, file=None):
        """Print the decision tree structure to console and/or file"""
        if node is None:
            node = self.root
        
        if node.predicted_class is not None:
            line = indent + f"Predict class: {node.predicted_class}"
            print(line)
            if file:
                file.write(line + "\n")
            return
        
        feature_name = f"feature_{node.feature_index}"
        if feature_names is not None:
            feature_name = feature_names[node.feature_index]
        
        if node.feature_type == 'numerical':
            condition = f"<= {node.threshold:.3f}"
        else:
            condition = f"== {node.threshold}"
            
        line = indent + f"{feature_name} {condition}"
        print(line)
        if file:
            file.write(line + "\n")
            
        line = indent + "├── True:"
        print(line)
        if file:
            file.write(line + "\n")
        self.print_tree(node.left_child, indent + "│   ", feature_names, file)
        
        line = indent + "└── False:"
        print(line)
        if file:
            file.write(line + "\n")
        self.print_tree(node.right_child, indent + "    ", feature_names, file)

    def visualize_tree(self, feature_names=None, filename="decision_tree_viz.png"):
        """Create a graphical visualization of the tree using matplotlib"""
        def plot_node(node, x, y, dx, dy, level, ax):
            if node.predicted_class is not None:
                ax.text(x, y, f"Class: {node.predicted_class}", ha='center', va='center',
                       bbox=dict(facecolor='lightgreen', edgecolor='black'))
                return
            
            feature_name = f"feature_{node.feature_index}"
            if feature_names is not None:
                feature_name = feature_names[node.feature_index]
                
            if node.feature_type == 'numerical':
                condition = f"{feature_name}\\n<= {node.threshold:.3f}"
            else:
                condition = f"{feature_name}\\n== {node.threshold}"
                
            # Draw current node
            ax.text(x, y, condition, ha='center', va='center',
                   bbox=dict(facecolor='lightblue', edgecolor='black'))
            
            # Calculate positions for children
            next_dx = dx / 2
            next_y = y - dy
            
            # Draw connections and recursive plot children
            if node.left_child:
                left_x = x - next_dx
                ax.plot([x, left_x], [y-0.1, next_y+0.1], 'k-')
                ax.text((x + left_x)/2, (y + next_y)/2, 'True', ha='center', va='bottom')
                plot_node(node.left_child, left_x, next_y, next_dx, dy, level+1, ax)
            
            if node.right_child:
                right_x = x + next_dx
                ax.plot([x, right_x], [y-0.1, next_y+0.1], 'k-')
                ax.text((x + right_x)/2, (y + next_y)/2, 'False', ha='center', va='bottom')
                plot_node(node.right_child, right_x, next_y, next_dx, dy, level+1, ax)
        
        plt.figure(figsize=(15, 10))
        ax = plt.gca()
        ax.set_axis_off()
        
        # Calculate initial spacing based on tree depth
        def get_depth(node):
            if node.predicted_class is not None:
                return 0
            return 1 + max(get_depth(node.left_child), get_depth(node.right_child))
        
        depth = get_depth(self.root)
        dy = 1.0 / (depth + 1)
        dx = 1.0
        
        # Plot the tree
        plot_node(self.root, 0.5, 1-dy, dx/4, dy, 0, ax)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

def visualize_tree_plot(node: TreeNode, feature_names: List[str] = None, depth: int = 0, 
                       pos: tuple = (0, 0), ax=None, parent_pos=None):
    """Create a matplotlib visualization of the tree."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
    # Draw node
    circle = plt.Circle(pos, 0.1, color='lightblue', fill=True)
    ax.add_patch(circle)
    
    # Add text for node
    if node.predicted_class is not None:
        text = f'Class: {node.predicted_class}'
    else:
        feature_name = (feature_names[node.feature_index] 
                       if feature_names is not None 
                       else f'Feature {node.feature_index}')
        if node.feature_type == 'numerical':
            text = f'{feature_name}\n<= {node.threshold:.2f}'
        else:
            text = f'{feature_name}\n== {node.threshold}'
    
    ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=8)
    
    # Draw line from parent if it exists
    if parent_pos is not None:
        ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')
    
    # Recursively visualize children
    if node.left_child is not None:
        left_pos = (pos[0] - 0.3/(depth+1), pos[1] - 0.2)
        visualize_tree_plot(node.left_child, feature_names, depth+1, 
                          left_pos, ax, pos)
    
    if node.right_child is not None:
        right_pos = (pos[0] + 0.3/(depth+1), pos[1] - 0.2)
        visualize_tree_plot(node.right_child, feature_names, depth+1, 
                          right_pos, ax, pos)
    
    return ax

def save_tree_visualization(tree: TreeNode, feature_names: List[str] = None, 
                          filename: str = 'decision_tree_viz.png'):
    """Save the tree visualization to a file."""
    plt.figure(figsize=(20, 10))
    ax = visualize_tree_plot(tree, feature_names)
    plt.axis('equal')
    plt.axis('off')
    plt.title('Decision Tree Visualization', pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Main function to demonstrate the decision tree implementation."""
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create feature types list (all numerical for breast cancer dataset)
    feature_types = ['numerical'] * X.shape[1]
    
    print("\nTraining decision tree...")
    decision_tree = DecisionTree(max_depth=5)  # Limit depth for better visualization
    decision_tree.fit(X_train, y_train, feature_names, feature_types)
    
    # Make predictions
    train_predictions = decision_tree.predict(X_train)
    test_predictions = decision_tree.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    # Print results to console
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nTree Structure:")
    print("==============\n")
    decision_tree.print_tree(feature_names=feature_names)

if __name__ == "__main__":
    main()
