from src.source import load_data
from src.transform import Transformer, balance_dataset
from src.train import train_model
from src.store import store_model
from metadata import MODEL_NAME


def main():
    df = load_data(file_name="Churn_Modelling_train_test.csv")
    df = balance_dataset(df=df, target="Exited")
    df = Transformer().transform(df)
    dt_model = train_model(df=df, target_column="Exited")
    store_model(model=dt_model, model_name=MODEL_NAME)


# This allows to run this code only when the main.py file is executed
# It won't be executed when importing it
if __name__ == "__main__":
    main()