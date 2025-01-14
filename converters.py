import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def numeric_converter(data):
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    data_norm = np.clip(data_norm, -1, 1)
    return data_norm, scaler
        