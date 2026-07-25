import yaml
import pandas as pd
import time
# sửa phần này 1 cách thống nhất để có thể nhúng các module một cách khoa học

from src.config import load_config, resolve_path, save
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
        self.results = []

    def run(self):
        # Lấy tất cả các block data
        n_blocks = cfg["split"]["n_blocks"]
        # n_blocks = 1 # set up fast test
        for block in range(n_blocks):
            
            #log
            print("============================================================")
            print("Current block:", block)

            pipeline = [self.flag_scale, self.flag_pca, self.n_components]

            # Lấy block data (i)
            X_train, y_train, X_test, y_test = load_block(block, cfg)

            # Tiền xử lí dữ liệu theo case cách set up thí nghiệm
            if self.flag_scale:
                preprocessor = Preprocess(flag_scale=True)
            elif self.flag_pca:
                preprocessor = Preprocess(flag_scale=True, flag_pca=True, n_components=self.n_components)
            else:
                preprocessor = Preprocess()
            
            start_preprocess = time.perf_counter()
            X_train_preprocessed, X_test_preprocessed = preprocessor.preprocess(X_train, X_test) 
            preprocess_time = time.perf_counter() - start_preprocess if self.flag_scale else 0.0
            print("Check preprocess:",X_train.sum(),X_train_preprocessed.sum()) # check xem data có được scale hay không

            # truyền data vào cho bộ train_test làm việc
            train_test = TrainTest([X_train_preprocessed, y_train],[X_test_preprocessed, y_test])
            train_test.run(block, pipeline, preprocess_time , self.results)
            
            # Các bước tiếp theo là tính toán d() để đưa ra báo cáo
        
    def log(self, file_name): #đưa các kết quả từ run() ra .csv/.txt

        path = resolve_path(cfg, 'results') / file_name
        df = pd.DataFrame(self.results)
        df.to_csv(
            path,
            mode="a",                     # append
            header=not path.exists(), # chỉ ghi header nếu file chưa tồn tại
            index=False,
        )

        