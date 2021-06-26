import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

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

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
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

    fig = plt.figure(figsize=(5 * num_params, 5 * num_scores))
    subfigs = fig.subfigures(num_scores, 1, squeeze=False, wspace=0.05, hspace=0.05)
    
    for j in range(num_scores):
        axes = subfigs[j,0].subplots(1, num_params, squeeze=False)
        subfigs[j,0].suptitle(f'{scores[j]} per Parameters')
        subfigs[j,0].supylabel('Best Score')
        
        for i, param in enumerate(params):
            plot_cv = pd.melt(
                cv_results, id_vars=[f'param_{param}'], 
                value_vars=[f'mean_train_{scores[j]}', f'mean_test_{scores[j]}'], 
                var_name='type', value_name=scores[j]
                )            
            try:
                sns.lineplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i])
            except TypeError:
                sns.violinplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i], palette="Set3")
            except ValueError:
                plot_cv[f'param_{param}'] = plot_cv[f'param_{param}'].map(str)
                sns.lineplot(x=f'param_{param}', y=scores[j], data=plot_cv, hue='type', ax=axes[0,i])
            except Exception:
                logger.exception(f'cannot plot search results for {scores[j]} {param}')
            
            axes[0, i].set_xlabel(param.upper())

    plt.subplots_adjust(top=0.92)
    return fig
