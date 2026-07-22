import yaml

# sửa phần này 1 cách thống nhất để có thể nhúng các module một cách khoa học

from src.config import load_config
from src.data.blocks import load_block
from src.pipelines.preprocess import Preprocess
from src.pipelines.train_test import TrainTest

cfg = load_config()

# Đọc yêu cầu từ thí nghiệm để thực hiện thí nghiệm: .run()
class BuildPipeline: 
    def __init__(self, flag_scale = False, flag_pca = False, n_components = None):

        self.flag_scale = flag_scale
        self.flag_pca = flag_pca
        self.n_components = n_components

    def run(self):
        # Lấy tất cả các block data
        n_blocks = cfg["split"]["n_blocks"]
        for i in range(n_blocks):
            
            print("============================================================")
            print("Current block:", i)
            
            # Lấy block data (i)
            X_train, y_train, X_test, y_test = load_block(i, cfg)

            # Tiền xử lí dữ liệu theo case cách set up thí nghiệm
            if self.flag_scale:
                preprocessor = Preprocess(flag_scale=True)
            elif self.flag_pca:
                preprocessor = Preprocess(flag_scale=True, flag_pca=True, n_components=self.n_components)
            else:
                preprocessor = Preprocess()
            X_train_preprocessed = preprocessor.preprocess(X_train)
            X_test_preprocessed = preprocessor.preprocess(X_test)

            # truyền data vào cho bộ train_test làm việc
            train_test = TrainTest([X_train_preprocessed, y_train],[X_test_preprocessed, y_test])
            train_test.run()
            
        
    # def log(): #đưa các kết quả từ run() ra .csv/.txt
        