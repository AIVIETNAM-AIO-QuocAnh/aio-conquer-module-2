import pandas as pd

from src.config import resolve_path,load_config
cfg = load_config()

def draw_figures():
    csv_path = resolve_path(cfg,'results') / 'PCA.csv'
    df = pd.read_csv(csv_path)  
    return df

df = draw_figures()
print(df.head())