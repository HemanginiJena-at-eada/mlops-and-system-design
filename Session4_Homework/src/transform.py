import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from metadata import (
    COLUMNS_TO_DROP,
    BINARY_FEATURES,
    ONE_HOT_ENCODE_COLUMNS,
)

#preprocessing features
class Transformer:
    def __init__(self):
        self.drop_columns = COLUMNS_TO_DROP
        self.binary_variable_columns = BINARY_FEATURES
        self.one_hot_encoding_columns = ONE_HOT_ENCODE_COLUMNS


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(self.drop_columns, axis=1)
        df = self._one_hot_encoding(df)
        
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
            transform="pandas"
        )
        encoder.fit(df[ONE_HOT_ENCODE_COLUMNS])
        encoded_df = encoder.transform(df[ONE_HOT_ENCODE_COLUMNS])
        df = df.drop(columns=ONE_HOT_ENCODE_COLUMNS)
        df = pd.concat([df, encoded_df], axis=1)

        return df

#balance the dataset
def balance_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Separate the classes
    df_y0 = df[df[target] == 0]
    df_y1 = df[df[target] == 1]

    # Find the smaller class size
    min_size = len(df_y1)

    # Randomly sample from each class
    df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

    # Concatenate back together
    df_balanced = pd.concat([df_y0_balanced, df_y1])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced