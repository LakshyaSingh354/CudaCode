import numpy as np

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Parameters:
        x (array-like): Input array (list, numpy array, etc.)
        
    Returns:
        numpy.ndarray: Softmax probabilities
    """
    # Convert input to numpy array
    x = np.array(x, dtype=np.float64)
    
    # Shift values by subtracting max (for numerical stability)
    shift_x = x - np.max(x)
    
    # Exponentiate values
    exp_x = np.exp(shift_x)
    print(np.sum(exp_x))
    
    # Normalize by sum
    return exp_x / np.sum(exp_x)

scores = [2.0, 1.0, 0.1]
print(softmax(scores))
