MODELS_FOLDER = "Session4_Homework/models"
DATASETS_FOLDER = "C:/Users/sudhi/Downloads/MLOps/Session 1_ML/mlops-and-system-design/datasets"
MODEL_NAME = "decision-tree-model"

COLUMNS_TO_DROP = ['RowNumber', 'CustomerId', 'Surname']
BINARY_FEATURES = []
ONE_HOT_ENCODE_COLUMNS = ['Geography', 'Gender']
MODEL_PARAMS = {
    "max_depth": 6,
    "min_samples_split": 10,
    "random_state": 21
}
