import numpy as np
import warnings 

def pielou_evenness(array, alpha = 0.5):
    """
    Compute Pielou's Evenness Index 
    
    0 : maximal skew of population towards clonal frequency of a single sequence
    1 : extreme evenness across clonal frequencies i
    
    Parameters
    ----------
    array: np.array
        array of values
    alpha : float 
        represents the smoothing parameter added to all values to avoid zero

    Returns
    -------
    p_evenness : float 
    
    between 0 and 1
    
    Examples 
    -------
    >>> pielou_evenness(np.array([1,1,1,1,1,1]))
    1.0
    >>> pielou_evenness(np.array([10,10,10,10,10,10]))
    1.0
    >>> pielou_evenness(np.array([10,1,1,1,1,1]))
    0.6546601218437399
    >>> pielou_evenness(np.array([10,1]))
    0.4394969869215134
    >>> pielou_evenness(np.array([1000,1,1,1,1,1]))
    0.02196415527496694
    >>> pielou_evenness(np.array([1]))
    1
    
    Notes
    -----
    Chiffelle et al. 2020: "Pielou's index, which is itself derived from the ratio between the Shannon entropy and 
    the maximization of the diversity distribution fo species with a sample...where '
    
    The compliment of clonal evenness (1-Pielou's index) is often used to get a Clonality score" 
        0 represents a maximally diverse population with even frequencies, and 
        1 represents a a set with a high degree of clonal dominance.

              N      
             ___     
             ╲      p   log  ⎛p ⎞
       =   - ╱       i     b ⎝ i⎠
             ‾‾‾           
             i=1       
                    ──────────────
                       log  (N) 
                           b   
    For N clones or species, indexed i, let p_i be the frequency of the ith clone
    
    
    Is sensitive to number of total clones. For more information see: 
    
    Chiffelle, J., Genolet, R., Perez, M. A., Coukos, G., Zoete, V., & Harari, A. (2020). 
    T-cell repertoire analysis and metrics of diversity and clonality. 
    Current Opinion in Biotechnology, 65, 284-295.
    """
    if array.size == 1:
        warnings.warn("pielou_evenness() received input array of length 1, returning 1 by default")
        return 1
    
    array = array.astype('float64').flatten()

    if np.any(array == 0):
        array += np.min(a[np.nonzero(a)])/2

    array = array / np.sum(array)
    
    numerator = np.dot(array, np.log2(array))
    denominator = np.log2(array.size)
    
    p_evenness =  -1 * (numerator/denominator)
    
    assert 0 <= p_evenness, "Pielou's index cannot be less than zero, check input values" 
    assert 1 >= p_evenness, "Pielou's index cannot be greater than 1, check input values"
    
    return p_evenness

