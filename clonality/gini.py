import numpy as np
def gini_index(array, alpha = 0.0000001 ):
    """
    Gini Index
    
    0 : maximal diversity  
    1 : extreme inequality in clonal frequency of a single sequence 
    
    Fast numpy implementation of Gini Index, modified from solution proposed by Olivia Guest, 
    with efficiency gain by placing values in ascending order.
    
    Parameters
    ----------
    
    array: np.array
        array of values, 
        negative values, cause a shift of values, such that the most negative value is zero
        zero values are smoothed by addition of alpha to all values 
    
    alpha : float 
        represents the smoothing parameter added to all values to avoid zero
    
    Calculate the Gini coefficient of a numpy array, adapted directed 
    by efficient solution by oliviaguest (https://github.com/oliviaguest/gini).
    
    See More at: 
    @misc{guest2017gini,
      author = "Olivia Guest",
      title = "Using the Gini Coefficient to Evaluate Deep Neural Network Layer Representations",
      year = "2017",
      howpublished = "Blog post",
      url = "http://neuroplausible.com/gini"
    }
    
    """
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    if np.any(array == 0):
        array += alpha  #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient