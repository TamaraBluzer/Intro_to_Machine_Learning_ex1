# 315287441 tamara bluzer
#211490362 itamar kolodny

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from typing import List, Union, Dict, Tuple

class TreeNode: # the class for the nodes in the decision tree

    def __init__(self, feature_index=None, threshold=None, feature_type=None,
                 left_child=None, right_child=None, predicted_class=None):
        self.feature_index = feature_index      # index of the feature to split on
        self.threshold = threshold              
        self.feature_type = feature_type        # numerical or categorical
        self.left_child = left_child           
        self.right_child = right_child         
        self.predicted_class = predicted_class  

class DecisionTree: # class of decision tree 
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None
        self.feature_types = None
    
    def entropy(self, y): # calculate the entropy of a node
        _, counts = np.unique(y, return_counts=True) #get probability of each class
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def information_gain(self, X, y, feature_idx, threshold, feature_type): #calcultes gain for a split
        parent_entropy = self.entropy(y)
        
        if feature_type == 'numerical':
            left_mask = X[:, feature_idx] <= threshold
        else:  # categorical
            left_mask = X[:, feature_idx] == threshold
            
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0: #if split is empty return 0
            return 0
        
        # calculate weighted entropy of children
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = n - n_left
        
        entropy_left = self.entropy(y[left_mask])
        entropy_right = self.entropy(y[right_mask])
        
        #calculate information gain
        weighted_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
        information_gain = parent_entropy - weighted_entropy
        
        return information_gain
    
    def find_best_split(self, X, y): #find the best feature and threshold to split
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_type = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_type = self.feature_types[feature_idx]
            unique_values = np.unique(X[:, feature_idx])
            
            if feature_type == 'numerical':
                if len(unique_values) > 1:
                    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                else:
                    continue
            else:  # categorical
                # try each unique value
                thresholds = unique_values
            
            for threshold in thresholds:   # try all possible thresholds
                gain = self.information_gain(X, y, feature_idx, threshold, feature_type)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_type = feature_type
        
        return best_feature, best_threshold, best_gain, best_type
    
    def build_tree(self, X, y, depth=0): #bulid the decision tree recursively 
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < 2:
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(predicted_class=leaf_value)
        
        best_feature, best_threshold, best_gain, best_type = self.find_best_split(X, y)
        
        if best_gain <= 0:  # if no improvement make a leaf node
            leaf_value = np.argmax(np.bincount(y))
            return TreeNode(predicted_class=leaf_value)
        
        if best_type == 'numerical':   # create child nodes based on feature type
            left_mask = X[:, best_feature] <= best_threshold
        else:  # categorical
            left_mask = X[:, best_feature] == best_threshold
            
        right_mask = ~left_mask
        
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(best_feature, best_threshold, best_type, left_tree, right_tree)
    
    def fit(self, X, y, feature_names=None, feature_types=None): #train the decision tree
        self.feature_names = feature_names
        self.feature_types = feature_types if feature_types else ['numerical'] * X.shape[1]
        self.root = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, node):
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
        return np.array([self.predict_sample(x, self.root) for x in X])
    
    def print_tree(self, node=None, indent="", feature_names=None, file=None):
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


def main():
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
