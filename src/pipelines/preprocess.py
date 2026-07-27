# Lấy block data từ root.data.blocks
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocess:
    def __init__(self, n_components = None, flag_scale = False, flag_pca = False):
        self.flag_scale = flag_scale
        self.flag_pca = flag_pca
        self.n_components = n_components
        self.time_preprocess_train = 0
        self.time_preprocess_test = 0
    def preprocess(self, X_train, X_test):
        if self.flag_scale or self.flag_pca:
            scaler = StandardScaler()

            start_scale_train = time.perf_counter()
            X_train = scaler.fit_transform(X_train)
            self.time_preprocess_train += time.perf_counter() - start_scale_train

            start_scale_test = time.perf_counter()
            X_test = scaler.transform(X_test)
            self.time_preprocess_test += time.perf_counter() - start_scale_test
        elif (self.flag_pca) and (self.n_components is not None):
            pca = PCA(n_components=self.n_components)

            start_pca_train = time.perf_counter()
            X_train = pca.fit_transform(X_train)
            self.time_preprocess_train += time.perf_counter() - start_pca_train

            start_pca_test = time.perf_counter()
            X_test = pca.transform(X_test)
            self.time_preprocess_test += time.perf_counter() - start_pca_test
        return [X_train, X_test, self.time_preprocess_train, self.time_preprocess_test]

