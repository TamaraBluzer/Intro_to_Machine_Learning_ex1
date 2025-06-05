import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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

def calculate_class_uncertainty(labels: np.ndarray) -> float:
    """Calculate the uncertainty (entropy) of a set of class labels."""
    # Calculate proportion of positive class
    positive_ratio = np.mean(labels)
    
    # Handle edge cases where all samples belong to one class
    if positive_ratio in [0, 1]:
        return 0
    
    # Calculate entropy for binary classification
    return -(positive_ratio * np.log2(positive_ratio) + 
            (1 - positive_ratio) * np.log2(1 - positive_ratio))

def calculate_split_quality(features: np.ndarray, labels: np.ndarray, 
                          feature_idx: int, threshold: Union[float, str], 
                          feature_type: str) -> float:
    """
    Calculate the quality of a split using information gain.
    Works for both numerical and categorical features.
    """
    if feature_type == 'numerical':
        # Split samples into left and right groups using <= for numerical
        left_mask = features[:, feature_idx] <= threshold
    else:  # categorical
        # Split samples into left and right groups using == for categorical
        left_mask = features[:, feature_idx] == threshold
        
    right_mask = ~left_mask
    
    # If split creates empty group, it's not valid
    if not np.any(left_mask) or not np.any(right_mask):
        return 0
    
    # Calculate proportions and entropies
    total_samples = len(labels)
    left_prop = np.sum(left_mask) / total_samples
    right_prop = 1 - left_prop
    
    parent_entropy = calculate_class_uncertainty(labels)
    left_entropy = calculate_class_uncertainty(labels[left_mask])
    right_entropy = calculate_class_uncertainty(labels[right_mask])
    
    # Calculate information gain
    information_gain = parent_entropy - (left_prop * left_entropy + right_prop * right_entropy)
    return information_gain

def find_optimal_split(features: np.ndarray, labels: np.ndarray, 
                      used_features: set, feature_types: List[str]) -> Tuple:
    """
    Find the best feature and threshold for splitting the data.
    Handles both numerical and categorical features.
    """
    best_gain = 0
    optimal_feature = None
    optimal_threshold = None
    optimal_type = None
    
    n_features = features.shape[1]
    
    # Try each feature that hasn't been used yet
    for feature_idx in range(n_features):
        if feature_idx in used_features:
            continue
            
        feature_type = feature_types[feature_idx]
        if feature_type == 'numerical':
            # For numerical features, try different thresholds
            unique_values = np.unique(features[:, feature_idx])
            
            # Use midpoints between consecutive values as thresholds
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                current_gain = calculate_split_quality(
                    features, labels, feature_idx, threshold, feature_type)
                
                if current_gain > best_gain:
                    best_gain = current_gain
                    optimal_feature = feature_idx
                    optimal_threshold = threshold
                    optimal_type = feature_type
                    
        else:  # categorical
            # For categorical features, try each unique category
            categories = np.unique(features[:, feature_idx])
            
            for category in categories:
                current_gain = calculate_split_quality(
                    features, labels, feature_idx, category, feature_type)
                
                if current_gain > best_gain:
                    best_gain = current_gain
                    optimal_feature = feature_idx
                    optimal_threshold = category
                    optimal_type = feature_type
                
    return optimal_feature, optimal_threshold, optimal_type

def construct_tree(features: np.ndarray, labels: np.ndarray, 
                  feature_types: List[str], used_features: set = None) -> TreeNode:
    """
    Recursively construct a decision tree.
    Handles both numerical and categorical features.
    """
    if used_features is None:
        used_features = set()
        
    # Base case 1: All samples have same label
    if np.all(labels == labels[0]):
        return TreeNode(predicted_class=labels[0])
    
    # Find the best split
    best_feature, best_threshold, feature_type = find_optimal_split(
        features, labels, used_features, feature_types)
    
    # Base case 2: No valid split found
    if best_feature is None:
        majority_class = 1 if np.mean(labels) >= 0.5 else 0
        return TreeNode(predicted_class=majority_class)
    
    # Track used features in this branch
    branch_features = used_features.union({best_feature})
    
    # Split data based on feature type
    if feature_type == 'numerical':
        left_mask = features[:, best_feature] <= best_threshold
    else:  # categorical
        left_mask = features[:, best_feature] == best_threshold
    
    right_mask = ~left_mask
    
    # Recursively build subtrees
    left_subtree = construct_tree(
        features[left_mask], labels[left_mask], feature_types, branch_features)
    right_subtree = construct_tree(
        features[right_mask], labels[right_mask], feature_types, branch_features)
    
    return TreeNode(
        feature_index=best_feature,
        threshold=best_threshold,
        feature_type=feature_type,
        left_child=left_subtree,
        right_child=right_subtree
    )

def predict_single_sample(tree_node: TreeNode, sample: np.ndarray) -> int:
    """Make a prediction for a single sample using the decision tree."""
    # Base case: reached a leaf node
    if tree_node.predicted_class is not None:
        return tree_node.predicted_class
    
    # Choose path based on feature type
    if tree_node.feature_type == 'numerical':
        # For numerical features, use <= comparison
        if sample[tree_node.feature_index] <= tree_node.threshold:
            return predict_single_sample(tree_node.left_child, sample)
        else:
            return predict_single_sample(tree_node.right_child, sample)
    else:
        # For categorical features, use == comparison
        if sample[tree_node.feature_index] == tree_node.threshold:
            return predict_single_sample(tree_node.left_child, sample)
        else:
            return predict_single_sample(tree_node.right_child, sample)

def predict_samples(samples: np.ndarray, tree: TreeNode) -> np.ndarray:
    """Predict class labels for multiple samples using the decision tree."""
    return np.array([predict_single_sample(tree, sample) for sample in samples])

def visualize_tree(node: TreeNode, depth: int = 0, feature_names: List[str] = None,
                  feature_types: List[str] = None) -> None:
    """Visualize the tree structure with feature names if available."""
    indent = "    " * depth
    
    if node.predicted_class is not None:
        print(f"{indent}Predict Class: {node.predicted_class}")
        return
    
    feature_name = (f"Feature '{feature_names[node.feature_index]}'" 
                   if feature_names is not None 
                   else f"Feature {node.feature_index}")
    
    if node.feature_type == 'numerical':
        print(f"{indent}Split on {feature_name} <= {node.threshold:.4f}")
    else:
        print(f"{indent}Split on {feature_name} == {node.threshold}")
    
    print(f"{indent}If True:")
    visualize_tree(node.left_child, depth + 1, feature_names, feature_types)
    print(f"{indent}If False:")
    visualize_tree(node.right_child, depth + 1, feature_names, feature_types)

if __name__ == "__main__":
    # Load breast cancer dataset (all numerical features)
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Define feature types (all numerical for breast cancer dataset)
    feature_types = ['numerical'] * X.shape[1]
    
    # Create a small example with both numerical and categorical features
    print("\nCreating example with mixed feature types...")
    X_mixed = np.column_stack([
        X[:, :2],  # Take first two numerical features
        np.random.choice(['A', 'B', 'C'], size=X.shape[0]),  # Add categorical feature
        np.random.choice(['Yes', 'No'], size=X.shape[0])     # Add another categorical feature
    ])
    feature_types_mixed = ['numerical', 'numerical', 'categorical', 'categorical']
    feature_names_mixed = ['mean radius', 'mean texture', 'category1', 'category2']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_mixed, y, test_size=0.2, random_state=42)
    
    # Train the tree
    print("\nTraining decision tree...")
    decision_tree = construct_tree(X_train, y_train, feature_types_mixed)
    
    # Visualize the tree
    print("\nDecision Tree Structure:")
    visualize_tree(decision_tree, feature_names=feature_names_mixed,
                  feature_types=feature_types_mixed)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predict_samples(X_test, decision_tree)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Show sample predictions
    print("\nSample predictions (first 5 test cases):")
    for i in range(5):
        print(f"True: {y_test[i]}, Predicted: {predictions[i]}")
        print("Feature values:")
        for j, (name, value) in enumerate(zip(feature_names_mixed, X_test[i])):
            print(f"  {name}: {value}")
