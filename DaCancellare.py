import numpy as np

def expand_feature_space(dataset):
    data = dataset[:-1]
    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size**2)
        return xxT
    expanded = np.apply_along_axis(vecxxT, 0, dataset)
    return np.vstack([expanded, dataset])

dataset = np.array([[1,2], [3,4], [5,6]])
prova = expand_feature_space(dataset)
print(5)