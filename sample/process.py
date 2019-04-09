import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ratio_satisfy(df, rat, random_split):
    good_df = df[df['label'] == 0]
    bad_df = df[df['label'] == 1]
    split_size = bad_df.shape[0] * rat / good_df.shape[0]
    gdf_less, gdf_testsize = train_test_split(
        good_df, test_size=split_size, random_state=random_split)
    df = pd.concat([gdf_testsize, bad_df])
    return df
