# Lấy block data từ root.data.blocks

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocess:
    def __init__(self, n_components = None, flag_scale = False, flag_pca = False):
        self.flag_scale = flag_scale
        self.flag_pca = flag_pca
        self.n_components = n_components
    def preprocess(self, data):
        if self.flag_scale:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        elif self.flag_pca:
            pca = PCA(n_components=self.n_components)
            data = pca.fit_transform(data)
        return data
