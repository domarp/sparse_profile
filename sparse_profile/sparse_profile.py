"""
Module to perform EDA tasks for a classification problem
with sparse data
@uthor : pramod.balakrishnan
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def sparse_profile(df, target_column, verbose = False):
    """
    Function to profile data

    Parameters
    ---------
    df : pandas dataframe, the data to be profiled (includes the target column)
    target_column : string, name of the target column in df
    verbose : boolean, print all output (default False)

    Returns
    -------
    ???
    """
    def compute_gain(class_col, pred_col, threshold):
        """
        Function to compute gain

        Parameters
        ----------
        class_col: pd.Series / np.array, target column
        pred_col: pd.Series / np.array, prediction column
        threshold: float, cut-off value for gain calculation

        Returns
        -------
        information gain
        """

        prior = np.mean(class_col)
        post_true = class_col[pred_col > threshold].mean()
        post_false = class_col[pred_col <= threshold].mean()
        weights = [(pred_col > threshold).sum(),(pred_col <= threshold).sum()]
        # print(weights)

        prior_entropy = -(prior*np.log(prior) + (1-prior)*np.log((1-prior)))

        modified_log = lambda x: np.log(x) if x > 0 else np.log(x+1e-10)

        post_true_entropy = post_true*modified_log(post_true) + (1-post_true)* modified_log((1-post_true))
        post_false_entropy = post_false*modified_log(post_false) + (1-post_false)*modified_log((1-post_false))

        post_entropy = -(weights[0]*post_true_entropy + weights[1]*post_false_entropy)/(weights[0]+weights[1])

        gain = -(post_entropy - prior_entropy)
        # print(threshold, prior_entropy, post_true_entropy,post_false_entropy, post_entropy, gain)

        return gain

    deciles = [i/10 for i in range(1,10)]
    no_target_df = df.drop([target_column], axis = 1)

    # report_sparsity to compute percentage of zeros
    report_sparsity = pd.DataFrame(1-no_target_df.replace(0, np.nan).count(axis = 0)/no_target_df.shape[0])
    if verbose:
        print("\nZeros (%)\n", report_sparsity)

    # distinct non zeros
    report_distinct = pd.DataFrame(no_target_df.apply(lambda x : len(np.unique(x)), axis=0))
    if verbose:
        print("\nDistinct Values (%)\n", report_distinct)

    # overall stats
    report_overall = pd.DataFrame(no_target_df.describe(percentiles=deciles))
    if verbose:
        print("\nOverall Stats\n", report_overall)

    # Non Zero stats
    report_non_zero = pd.DataFrame(no_target_df.replace(0, np.nan).describe(percentiles=deciles))
    if verbose:
        print("\nNon zero stats\n", report_non_zero)

    # Compute gain
    indx = [str(int(i*100)) + "%" for i in deciles]
    gain_df = pd.DataFrame(index = indx, columns  = report_overall.columns)
    auc_df = pd.DataFrame(columns  = report_overall.columns)

    for cols in report_overall.columns:
        temp_col = []
        for decile in indx:
            temp_col.append(compute_gain(df[target_column], df[cols], report_overall.loc[decile, cols]))
        gain_df[cols] = temp_col
        auc_df[cols] = [roc_auc_score(df[target_column], df[cols])]
    
    if verbose:
        print("\nGain DF\n", gain_df)
    
    print("\n Top features:\n")
    top_gain = gain_df.max().sort_values(ascending = False)
    print(top_gain)
    
    return (report_sparsity, report_distinct, report_overall, report_non_zero, gain_df, auc_df)


if __name__ == "__main__":
    df = pd.DataFrame({
        'target' : [1, 1, 1, 1, 0, 0 ,0 ,0, 1, 0],
        'col_1' :  [1, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        'col_2' :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    sparse_profile(df, 'target', False)


    



