# helper functions file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score

def model_eval(actual, pred):
    ''' Take in two series of pred and actual and
        calculate a variety of evaluation metrics'''
    auc = roc_auc_score(actual, pred)
    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    cm = confusion_matrix(actual, pred)
    recall = cm[1, 1]/(cm[1, 1] + cm[1, 0])
    print(f'AUC Score: {auc}')
    print(f'Accuracy Score: {acc}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print(cm)
    return auc, acc, f1, recall, cm


def pooled_var(stds):
    n = len(stds) # size of each group
    return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))


def add_model_scores_to_results(file_path, model_name:str, datashift:str, with_sw:int,
                                ROC_AUC, accuracy, f1, recall, cm:np.ndarray, first_entry=False):
    '''Take in metrics of a testing set and append to model metrics .csv file'''

    results_df = pd.DataFrame(columns=['Timestamp', 'model_name', 'balanced', 
                                     'with_sw', 'ROC_AUC', 'accuracy', 'F1_score', 'recall', 'cm'])
    results_df.loc[0] = [pd.Timestamp.now() , model_name, datashift, 
                         with_sw, ROC_AUC, accuracy, f1, recall, cm.ravel()]

    results_df.to_csv(path_or_buf=file_path, mode='a', header=first_entry, index=False)

    print(f'{model_name} saved successfully!')


def metric_cv_plot_singleParam(cv_results, metric, param_dict):
    results = [f'mean_test_{metric}', f'std_test_{metric}',
               f'mean_train_{metric}', f'std_train_{metric}']
    fig, axes = plt.subplots(1, len(param_dict), 
                         figsize = (3*len(param_dict), 4),
                         sharey='row')
    axes.set_ylabel(metric, fontsize=18)
    lw = 2

    for idx, (param_name, param_range) in enumerate(param_dict.items()):
        grouped_df = cv_results.groupby(f'param_{param_name}')[results]\
            .agg({f'mean_test_{metric}': 'mean',
                  f'std_test_{metric}': pooled_var,
                  f'mean_train_{metric}': 'mean',
                  f'std_train_{metric}': pooled_var})

        axes.set_xlabel(f'L2 param: {param_name}', fontsize=15)
        axes.set_ylim(0.0, 1.1)
        axes.plot(param_range, 
                    grouped_df[results[0]],
                    label=f"Cross-val {metric}",
                    color="darkorange",
                    lw=lw, marker='o')
        axes.fill_between(param_range,
                    grouped_df[results[0]] - grouped_df[results[3]],
                    grouped_df[results[0]] + grouped_df[results[3]],
                    alpha=0.2,
                    color="darkorange",
                    lw=lw)
        axes.plot(param_range, 
                    grouped_df[results[2]],
                    label=f"Traning split {metric}",
                    color="navy",
                    lw=lw, marker='o')
        axes.fill_between(param_range,
                    grouped_df[results[2]] - grouped_df[results[3]],
                    grouped_df[results[2]] + grouped_df[results[3]],
                    alpha=0.2,
                    color="navy",
                    lw=lw)

    handles, labels = axes.get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=15)
    fig.legend(handles, labels, fontsize=10, bbox_to_anchor=(1.55, 0.87), borderaxespad=0)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    #plt.show()
    plt.tight_layout()
    return fig


def metric_cv_plot_multiParam(cv_results, metric, param_dict):
    results = [f'mean_test_{metric}', f'std_test_{metric}',
               f'mean_train_{metric}', f'std_train_{metric}'] 
    fig, axes = plt.subplots(1, len(param_dict), 
                         figsize = (3*len(param_dict), 4),
                         sharey='row')
    axes[0].set_ylabel(metric, fontsize=18)
    lw = 2

    for idx, (param_name, param_range) in enumerate(param_dict.items()):
        grouped_df = cv_results.groupby(f'param_{param_name}')[results]\
            .agg({f'mean_test_{metric}': 'mean',
                  f'std_test_{metric}': pooled_var,
                  f'mean_train_{metric}': 'mean',
                  f'std_train_{metric}': pooled_var})

        axes[idx].set_xlabel(f'Parameter: {param_name}', fontsize=15)
        axes[idx].set_ylim(0.0, 1.1)
        axes[idx].plot(param_range, 
                    grouped_df[results[0]],
                    label=f"Cross-val {metric}",
                    color="darkorange",
                    lw=lw, marker='o')
        axes[idx].fill_between(param_range,
                    grouped_df[results[0]] - grouped_df[results[3]],
                    grouped_df[results[0]] + grouped_df[results[3]],
                    alpha=0.2,
                    color="darkorange",
                    lw=lw)
        axes[idx].plot(param_range, 
                    grouped_df[results[2]],
                    label=f"Traning split {metric}",
                    color="navy",
                    lw=lw, marker='o')
        axes[idx].fill_between(param_range,
                    grouped_df[results[2]] - grouped_df[results[3]],
                    grouped_df[results[2]] + grouped_df[results[3]],
                    alpha=0.2,
                    color="navy",
                    lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=15)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=10)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    #plt.show()
    return fig


def random_dropout(sentence, p, random_state):
    '''Perform random word dropout on a sentence with probability, p'''
    rng = np.random.default_rng(seed=random_state)
    tokens = sentence.split()
    num_words = len(tokens)
    mask = rng.binomial(1, p, num_words)
    keep_tok = [tok for tok, b in zip(tokens, mask) if b]
    dropped_tokens = ' '.join(keep_tok)
    return dropped_tokens


def get_wordcount(sentence):
    return len(sentence.split())


