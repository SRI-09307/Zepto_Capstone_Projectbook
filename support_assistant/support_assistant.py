
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "..",
    "data_pipeline",
    "books_data.csv"
)

df = pd.read_csv(DATA_FILE)

print(df.head())