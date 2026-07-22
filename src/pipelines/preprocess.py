# Lấy block data từ root.data.blocks

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocess:
    def __init__(self, n_components = None, flag_scale = False, flag_pca = False):
        self.flag_scale = flag_scale
        self.flag_pca = flag_pca
        self.scaler = StandardScaler()
        self.n_components = n_components
    def scaling(self, data):
        if self.flag_scale:
            data_scaled = self.scaler.fit_transform(data)
        return data_scaled
    def pca(self, n_components, data):
        if(self.flag_pca):
            self.n_components = n_components
            data_scaled = self.scaler.fit_transform(data)
            pca = PCA(n_components=self.n_components)
            data_pca = pca.fit_transform(data_scaled)
        return data_pca
    def preprocess(self, data):
        if self.flag_scale:
            data_scaled = self.scaling(data)
            return data_scaled
        elif self.flag_pca:
            data_pca = self.pca(self.n_components, data)
            return data_pca
        else :
            return data
