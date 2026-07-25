# khởi tạo grid-params -> train (.fit) + valid -> test
# Train + hyperparameter tuning + evaluation
import time


from src.config import load_config, save
cfg = load_config()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.models import get_model
model_keys = ["KNeighborsClassifier", "LogisticRegression", "RandomForestClassifier"]

class TrainTest:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
    def run(self, block, pipeline, preprocess_time, results):
        for model_key in model_keys:
            print(f"    Current model: {model_key}")
            result = {}
            save(result, 'block_id', block)
            save(result, 'pipeline', pipeline)
            save(result, 'model', model_key)

            # Train
            model = get_model(model_key, random_state=cfg['reproducibility']['random_seed'])

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=cfg["models"][model_key]["param_grid"],
                cv=cfg['grid_search']['k_fold'],
                scoring=cfg['grid_search']['scoring']
            )

            start_time = time.perf_counter()
            grid_search.fit(self.train_data[0], self.train_data[1])
            fit_runtime_sec = time.perf_counter() - start_time


            print("     Best hyperparams:", grid_search.best_params_)
            print("     Best Score:", grid_search.best_score_)
            save(result, 'fit_runtime_sec', fit_runtime_sec)
            save(result, 'best_params', grid_search.best_params_)

            # Test
            start_infer = time.perf_counter()
            y_pred = grid_search.predict(self.test_data[0]) #[0] = X_test_preprocessed
            accuracy = accuracy_score(self.test_data[1], y_pred) #[1] = y_test
            macro_f1 = f1_score(self.test_data[1], y_pred, average='macro')
            inference_runtime_sec = time.perf_counter() - start_infer + preprocess_time

            save(result, 'accuracy', accuracy)
            save(result, 'macro_f1', macro_f1)
            save(result, 'inference_runtime_sec', inference_runtime_sec)

            results.append(result)