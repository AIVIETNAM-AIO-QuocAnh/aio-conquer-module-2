# khởi tạo grid-params -> train (.fit) + valid -> test
# Train + hyperparameter tuning + evaluation

from src.config import load_config
cfg = load_config()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models import get_model
model_keys = ["KNeighborsClassifier", "LogisticRegression", "RandomForestClassifier"]

class TrainTest:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def run(self):
        for model_key in model_keys:
            print(f"    Current model: {model_key}")

            # Train
            model = get_model(model_key, random_state=cfg['reproducibility']['random_seed'])
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=cfg["models"][model_key]["param_grid"],
                cv=cfg['grid_search']['k_fold'],
                scoring=cfg['grid_search']['scoring']
            )
            grid_search.fit(self.train_data[0], self.train_data[1])

            print("     Best hyperparams:", grid_search.best_params_)
            print("     Best Score:", grid_search.best_score_)

            # Test
            predictions = grid_search.predict(self.test_data[0])
