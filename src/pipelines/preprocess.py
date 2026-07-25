# Lấy block data từ root.data.blocks

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocess:
    def __init__(self, n_components = None, flag_scale = False, flag_pca = False):
        self.flag_scale = flag_scale
        self.flag_pca = flag_pca
        self.n_components = n_components
    def preprocess(self, X_train, X_test):
        if self.flag_scale or self.flag_pca:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif (self.flag_pca) and (self.n_components is not None):
            pca = PCA(n_components=self.n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        return [X_train, X_test]

