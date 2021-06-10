import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_features(data, sents_col, keys, feature_func):
    df = feature_func(data[sents_col])
    features = df.select_dtypes(exclude=object).columns
    keys_col = data[keys]
    df = df.join(keys_col)

    num_feat = len(features)
    num_keys = len(keys)
    if num_keys == 1:
        keys = [keys]

    fig = plt.figure(figsize=(5 * num_keys, 4 * num_feat))
    subfigs = fig.subfigures(num_feat, 1, squeeze=False, wspace=0.05, hspace=0.05)

    for j in range(num_feat):
        axes = subfigs[j,0].subplots(1, num_keys, squeeze=False)
        subfigs[j,0].suptitle(f'{features[j]} Comparison')

        for i, key in enumerate(keys):
            sns.histplot(data=df, x=features[j], hue=key, ax=axes[0,i])
    
    plt.show()
