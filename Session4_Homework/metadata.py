from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

DATASETS_FOLDER = CURRENT_DIR.parent / "datasets"

MODELS_FOLDER = "Session4_Homework/models"

MODEL_NAME = "decision-tree-model"

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
BINARY_FEATURES = []
ONE_HOT_ENCODE_COLUMNS = ["Geography", "Gender"]
MODEL_PARAMS = {"max_depth": 6, "min_samples_split": 10, "random_state": 21}
