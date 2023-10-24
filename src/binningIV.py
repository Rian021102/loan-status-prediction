import pandas as pd
import numpy as np

def create_binning(data, predictors, num_of_bins):
    data [predictors+"_bin"] =pd.qcut(data[predictors], q=num_of_bins,
    duplicates='drop')
    return data


