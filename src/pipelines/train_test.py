# khởi tạo grid-params -> train (.fit) + valid -> test
# Train + hyperparameter tuning + evaluation
from src.config import load_config
cfg = load_config()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


model_classes = {
    'KNeighborsClassifier' : KNeighborsClassifier,
    'LogisticRegression' : LogisticRegression,
    'RandomForestClassifier' : RandomForestClassifier
}

# Để đẩy model cho train_test cần việc của AI engineer (model)
class TrainTest:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def run(self):
        for model_name, model in model_classes.items():
            print(f"    Current model: {model_name}")

            # Train
            cv_model = GridSearchCV(
                estimator=model(),
                param_grid=cfg['hyperparam_space'][model_name],
                cv=cfg['cv']
            )
            cv_model.fit(self.train_data[0], self.train_data[1])

            print("     Best hyperparams:", cv_model.best_params_)

            # Test
            predictions = cv_model.predict(self.test_data[0])
