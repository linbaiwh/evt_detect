import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_features(data, sents_col, keys, feature_func, **kwargs):
    df = feature_func(data[sents_col], **kwargs)
    features = df.select_dtypes(exclude=object).columns
    keys_col = data[keys]
    df = df.join(keys_col)

    num_feat = len(features)
    num_keys = len(keys)

    fig = plt.figure(figsize=(5 * num_keys, 4 * num_feat))
    subfigs = fig.subfigures(num_feat, 1, squeeze=False, wspace=0.05, hspace=0.05)

    for j in range(num_feat):
        axes = subfigs[j,0].subplots(1, num_keys, squeeze=False)
        subfigs[j,0].suptitle(f'{features[j]} Comparison')

        for i, key in enumerate(keys):
            sns.violinplot(data=df, x=key, y=features[j], ax=axes[0,i])
    
    plt.show()

def plot_search_results(grid):
    """Plot training/validation scores against hyperparameters

    Args:
        grid (GridSearchCV): GridSearchCV Instance that have cv_results
    """
    cv_results = pd.DataFrame(grid.cv_results_)
    params = [param[6:] for param in cv_results.columns if 'param_' in param and cv_results[param].nunique() > 1]
    num_params = len(params)
    scores = [score[11:] for score in cv_results.columns if 'mean_train_' in score]
    num_scores = len(scores)

    fig = plt.figure(figsize=(5 * num_params, 4 * num_scores))
    subfigs = fig.subfigures(num_scores, 1, squeeze=False, wspace=0.05, hspace=0.05)
    
    for j in range(num_scores):
        axes = subfigs[j,0].subplots(1, num_params, squeeze=False)
        subfigs[j,0].suptitle(f'{scores[j]} per Parameters')
        subfigs[j,0].supylabel('Best Score')
        
        for i, param in enumerate(params):
            param_group = cv_results.groupby(f'param_{param}')
            x = param_group.groups.keys()
            means_train = param_group[f'mean_train_{scores[j]}'].min()
            e_2 = cv_results.loc[param_group.idxmin()[f'mean_train_{scores[j]}'], f'std_train_{scores[j]}']
            means_test = param_group[f'mean_test_{scores[j]}'].min()
            e_1 = cv_results.loc[param_group.idxmin()[f'mean_test_{scores[j]}'], f'std_test_{scores[j]}']
            
            axes[0, i].errorbar(x, means_test, e_1, linestyle='--', marker='o', label='test')
            axes[0, i].errorbar(x, means_train, e_2, linestyle='-', marker='^',label='train' )
            
            axes[0, i].set_xlabel(param.upper())
            axes[0,i].legend()

    plt.subplots_adjust(top=0.9)
    plt.legend()
    plt.show()
