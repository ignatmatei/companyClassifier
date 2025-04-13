import matplotlib.pyplot as plt
import numpy as np

def printStringClassifier(results) :
    i = 0
    x_axis = []
    y_axis = []
    while i < 2 :
     x_axis.append(i)
     results = {key : value for key, value in results.items() if value[1] > i}
     y_axis.append(len(results))
     i += 0.1

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, y_axis, color='orange', linestyle='-', marker='o')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Number of classified items', fontsize=12)
    plt.savefig('threshold_plot.pdf', format='pdf')

def print_label_statistics(X, y, num_classes):
    """
    Print statistics about the labeled data.
    
    Args:
        X (list): List of company texts
        y (list): List of labels
        num_classes (int): Number of unique labels
    """
    print(f"\nNumber of companies with labels: {len(X)}")
    print(f"Number of unique labels: {num_classes}")
    print("\nLabel distribution:")
    
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} companies")
        
    # Calculate and print percentage distribution
    total = sum(counts)
    print("\nPercentage distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / total) * 100
        print(f"{label}: {percentage:.2f}%")
