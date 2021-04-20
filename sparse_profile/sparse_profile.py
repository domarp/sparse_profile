"""
Module to perform EDA tasks for a classification problem
with sparse data
@uthor : pramod.balakrishnan
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class sparse_profile:
    def compute_gain(self, pred_column, threshold):
        """
        Function to compute gain

        Parameters
        ----------
        ???

        Returns
        -------
        information gain
        """

        class_col = self.y.values
        pred_col = self.x[pred_column]

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

    def sparse_profile_report(self):
        """
        Function to profile data

        Parameters
        ---------
        ???

        Returns
        -------
        ???
        """
        
        deciles = [i/10 for i in range(1,10)]
        no_target_df = self.x.copy()

        # report_sparsity to compute percentage of zeros
        self.report_sparsity = pd.DataFrame(1-no_target_df.replace(0, np.nan).count(axis = 0)/no_target_df.shape[0])
        if self.verbose:
            print("\nZeros (%)\n", self.report_sparsity)

        # distinct non zeros
        self.report_distinct = pd.DataFrame(no_target_df.apply(lambda x : len(np.unique(x)), axis=0))
        if self.verbose:
            print("\nDistinct Values (%)\n", self.report_distinct)

        # overall stats
        self.report_overall = pd.DataFrame(no_target_df.describe(percentiles=deciles))
        if self.verbose:
            print("\nOverall Stats\n", self.report_overall)

        # Non Zero stats
        self.report_non_zero = pd.DataFrame(no_target_df.replace(0, np.nan).describe(percentiles=deciles))
        if self.verbose:
            print("\nNon zero stats\n", self.report_non_zero)

        # Compute gain
        indx = [str(int(i*100)) + "%" for i in deciles]
        self.gain_df = pd.DataFrame(index = indx, columns  = self.report_overall.columns)
        self.auc_df = pd.DataFrame(columns  = self.report_overall.columns)

        for cols in self.report_overall.columns:
            temp_col = []
            for decile in indx:
                temp_col.append(self.compute_gain(cols, self.report_overall.loc[decile, cols]))
            self.gain_df[cols] = temp_col
            self.auc_df[cols] = [roc_auc_score(self.y, self.x[cols])]
        
        if self.verbose:
            print("\nGain DF\n", self.gain_df)
        
        print("\n Top features:\n")
        self.top_gain = self.gain_df.max().sort_values(ascending = False)
        print(self.top_gain)

    def __init__(self, df, target_column, verbose=False):
        self.y = df[target_column]
        self.x = df.drop([target_column], axis=1)
        self.verbose = False
        self.sparse_profile_report()

if __name__ == "__main__":
    df = pd.DataFrame({
        'target' : [1, 1, 1, 1, 0, 0 ,0 ,0, 1, 0],
        'col_1' :  [1, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        'col_2' :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    sProfile = sparse_profile(df, 'target')
    print(sProfile.top_gain)


    



