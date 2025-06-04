import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def entropy(y):
    """Calculate the entropy of a binary array."""
    q = y.sum() / y.size
    if q == 0 or q == 1:  # Avoid log(0)
        return 0
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)

def information_gain(X, y, feature_index, threshold):
    """Calculate the information gain for a given feature and threshold."""
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold

    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
        return 0  # Avoid meaningless splits

    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])

    p_left = np.sum(left_indices) / len(y)
    p_right = np.sum(right_indices) / len(y)

    return entropy(y) - (p_left * left_entropy + p_right * right_entropy)

def best_split(X, y, used_features):
    """Find the best feature and threshold to split the data, avoiding used features."""
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        if feature_index in used_features:
            continue  # Skip already used features

        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, used_features=set()):
    """Recursively build the decision tree, avoiding reused features in the same branch."""
    # Base case: If all labels are the same, create a leaf node
    if np.all(y == y[0]):
        return Node(value=y[0])

    # Find the best feature and threshold for splitting
    feature, threshold = best_split(X, y, used_features)
    if feature is None:  # No valid split found
        most_common = 1 if y.sum() > y.size / 2 else 0
        return Node(value=most_common)

    # Update the set of used features for this branch
    used_features = used_features.union({feature})

    # Split the data into left and right branches
    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold

    # Recursively build left and right subtrees
    left = build_tree(X[left_indices], y[left_indices], used_features)
    right = build_tree(X[right_indices], y[right_indices], used_features)

    return Node(feature=feature, threshold=threshold, left=left, right=right)

def predict_tree(node, x):
    """Make a prediction for a single sample using the decision tree."""
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

def predict(X, tree):
    """Predict for multiple samples using the decision tree."""
    return np.array([predict_tree(tree, x) for x in X])

def print_tree(node, depth=0):
    """Recursively print the tree structure."""
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: Class {node.value}")
    else:
        print(f"{indent}Feature {node.feature} <= {node.threshold}")
        print(f"{indent}Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}Right:")
        print_tree(node.right, depth + 1)

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features")

    # Split into train and test sets
    print("\nSplitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Build the tree using the training set
    print("\nBuilding decision tree...")
    tree = build_tree(X_train, y_train)

    # Print the tree structure
    print("\nDecision Tree Structure:")
    print_tree(tree)

    # Predict on the test set
    print("\nMaking predictions...")
    predictions = predict(X_test, tree)

    # Print predictions
    print("\nFirst 10 predictions:", predictions[:10])
    print("First 10 actual values:", y_test[:10])

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
