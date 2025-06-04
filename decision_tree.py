import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class TreeNode:
    """A class representing a node in the decision tree"""
    def __init__(self, feature_index=None, threshold=None, left_child=None, 
                 right_child=None, predicted_class=None):
        self.feature_index = feature_index      # Index of the feature to split on
        self.threshold = threshold              # Threshold value for the split
        self.left_child = left_child           # Left subtree
        self.right_child = right_child         # Right subtree
        self.predicted_class = predicted_class  # Class prediction (for leaf nodes)

def calculate_class_uncertainty(labels):
    """
    Calculate the uncertainty (entropy) of a set of class labels.
    For binary classification: -p(0)log2(p(0)) - p(1)log2(p(1))
    """
    # Calculate proportion of positive class
    positive_ratio = np.mean(labels)
    
    # Handle edge cases where all samples belong to one class
    if positive_ratio in [0, 1]:
        return 0
    
    # Calculate entropy for binary classification
    return -(positive_ratio * np.log2(positive_ratio) + 
            (1 - positive_ratio) * np.log2(1 - positive_ratio))

def calculate_split_quality(features, labels, feature_idx, split_value):
    """
    Calculate the quality of a split using information gain.
    Higher values indicate better splits.
    """
    # Split samples into left and right groups
    left_mask = features[:, feature_idx] <= split_value
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

def find_optimal_split(features, labels, used_features):
    """
    Find the best feature and threshold for splitting the data.
    Avoids reusing features that were already used in the current branch.
    """
    best_gain = 0
    optimal_feature = None
    optimal_threshold = None
    
    n_features = features.shape[1]
    
    # Try each feature that hasn't been used yet
    for feature_idx in range(n_features):
        if feature_idx in used_features:
            continue
            
        # Get unique values for potential thresholds
        unique_values = np.unique(features[:, feature_idx])
        
        # Try each threshold
        for threshold in unique_values:
            current_gain = calculate_split_quality(features, labels, 
                                                feature_idx, threshold)
            
            # Update best split if current is better
            if current_gain > best_gain:
                best_gain = current_gain
                optimal_feature = feature_idx
                optimal_threshold = threshold
                
    return optimal_feature, optimal_threshold

def construct_tree(features, labels, used_features=None):
    """
    Recursively construct a decision tree using binary splitting.
    """
    if used_features is None:
        used_features = set()
        
    # Base case 1: All samples have same label
    if np.all(labels == labels[0]):
        return TreeNode(predicted_class=labels[0])
    
    # Find the best split
    best_feature, best_threshold = find_optimal_split(features, labels, used_features)
    
    # Base case 2: No valid split found
    if best_feature is None:
        majority_class = 1 if np.mean(labels) >= 0.5 else 0
        return TreeNode(predicted_class=majority_class)
    
    # Track used features in this branch
    branch_features = used_features.union({best_feature})
    
    # Split data
    left_mask = features[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    # Recursively build subtrees
    left_subtree = construct_tree(features[left_mask], labels[left_mask], 
                                branch_features)
    right_subtree = construct_tree(features[right_mask], labels[right_mask], 
                                 branch_features)
    
    return TreeNode(best_feature, best_threshold, left_subtree, right_subtree)

def predict_single_sample(tree_node, sample):
    """Make a prediction for a single sample using the decision tree."""
    # Base case: reached a leaf node
    if tree_node.predicted_class is not None:
        return tree_node.predicted_class
    
    # Traverse left or right based on the split
    if sample[tree_node.feature_index] <= tree_node.threshold:
        return predict_single_sample(tree_node.left_child, sample)
    else:
        return predict_single_sample(tree_node.right_child, sample)

def predict_samples(samples, tree):
    """Predict class labels for multiple samples using the decision tree."""
    return np.array([predict_single_sample(tree, sample) for sample in samples])

def visualize_tree(node, depth=0, feature_names=None):
    """Visualize the tree structure with feature names if available."""
    indent = "    " * depth
    
    if node.predicted_class is not None:
        print(f"{indent}Predict Class: {node.predicted_class}")
        return
        
    feature_name = (f"Feature '{feature_names[node.feature_index]}'" 
                   if feature_names is not None 
                   else f"Feature {node.feature_index}")
    
    print(f"{indent}Split on {feature_name} <= {node.threshold:.4f}")
    print(f"{indent}If True:")
    visualize_tree(node.left_child, depth + 1, feature_names)
    print(f"{indent}If False:")
    visualize_tree(node.right_child, depth + 1, feature_names)

if __name__ == "__main__":
    # Load and prepare the breast cancer dataset
    print("Loading breast cancer dataset...")
    cancer_data = load_breast_cancer()
    X, y = cancer_data.data, cancer_data.target
    feature_names = cancer_data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train the decision tree
    print("\nTraining decision tree...")
    decision_tree = construct_tree(X_train, y_train)
    
    # Visualize the trained tree
    print("\nDecision Tree Structure:")
    visualize_tree(decision_tree, feature_names=feature_names)
    
    # Make predictions
    print("\nEvaluating model...")
    train_predictions = predict_samples(X_train, decision_tree)
    test_predictions = predict_samples(X_test, decision_tree)
    
    # Calculate and display metrics
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print(f"Testing Accuracy: {test_accuracy:.2%}")
    
    # Display sample predictions
    print("\nSample predictions (first 10 test cases):")
    for i in range(10):
        print(f"True: {y_test[i]}, Predicted: {test_predictions[i]}")
