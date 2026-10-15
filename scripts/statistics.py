import matplotlib.pyplot as plt
import pandas as pd

def z_score(series) -> pd.Series:
    """Calculate the z-score of a pandas Series."""
    return (series - series.mean()) / series.std()

def z_score_diff(series1, series2)-> pd.Series:
    """Calculate the z-score of the difference of a pandas Series."""
    return z_score(series1)- z_score(series2)

