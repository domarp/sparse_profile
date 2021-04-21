"""
@uthor : pramod.balakrishnan

Module to perform EDA tasks for a classification problem
with sparse data
Curently takes only numeric values

Sample usage
------------
df = pd.DataFrame({
        'target' : [1, 1, 1, 1, 0, 0 ,0 ,0, 1, 0],
        'col_1' :  [1, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        'col_2' :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    sProfile = sparse_profile(df, 'target')
    print(sProfile.top_gain)

Various sparse_profile reports can be accessed as attributes of the 
sparse_profile class object

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
        self:        sparse_profile class object
        pred_column: string, column name of the column being evaluated / gain being computed
        threshold:   float, threshold value at which gain is to be computed

        Returns
        -------
        information gain of pred_column at threshold for the target column
        """

        class_col = self.y.values
        pred_col = self.x[pred_column]

        prior = np.mean(class_col)
        post_true = class_col[pred_col > threshold].mean()
        post_false = class_col[pred_col <= threshold].mean()
        weights = [(pred_col > threshold).sum(),(pred_col <= threshold).sum()]

        prior_entropy = -(prior*np.log(prior) + (1-prior)*np.log((1-prior)))

        modified_log = lambda x: np.log(x) if x > 0 else np.log(x+1e-10)

        post_true_entropy = post_true*modified_log(post_true) + (1-post_true)* modified_log((1-post_true))
        post_false_entropy = post_false*modified_log(post_false) + (1-post_false)*modified_log((1-post_false))

        post_entropy = -(weights[0]*post_true_entropy + weights[1]*post_false_entropy)/(weights[0]+weights[1])

        gain = -(post_entropy - prior_entropy)

        return gain

    def sparse_profile_report(self):
        """
        Function to create sparse profile report

        Parameters
        ---------
        self: sparse_profile class object

        Returns
        -------
        Sparse profile report in terms of corresponding class object attributes
        report_sparsity:    pandas dataframe, Percentage of zeros in each column
        report_distinct:    pandas dataframe, Count of distinct non zero values in each column
        report_overall:     pandas dataframe, Overall summary of each column
                            (similar to pandas describe())
        report_non_zero:    pandas dataframe, Summary of each column after removing zeros
        gain_df:            pandas dataframe, Relative information gain at decile cutoffs 
                            for each column wrt target column
        auc_df:             pandas dataframe, AUC of each column wrt target column
        top_gain:           pandas dataframe, Columns sorted by maximum gain obtained from gain_df
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
        
        # print("\n Top features:\n")
        self.top_gain = self.gain_df.max().sort_values(ascending = False)
        # print(self.top_gain)

    def __init__(self, df, target_column, verbose=False):
        """
        Initialization of class object

        Parameters
        ----------
        df:             pandas dataframe, contains target and predictor columns (all numeric)
        target_column:  string, target column name
        verbose:        boolean, to print completion status

        Returns
        -------
        sparse_profile class object initialized to values passed to the initialization function
        and sparse_profile reports computed
        """

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


    



