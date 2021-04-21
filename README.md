# Sparse profile  - EDA on sparse data
Module to perform EDA tasks for a classification problem with sparse data<br>
Curently takes only numeric values

Sample usage
------------
```python
import pandas as pd
import numpy as np
from sparse_profile import sparse_profile

df = pd.DataFrame({
        'target' : [1, 1, 1, 1, 0, 0 ,0 ,0, 1, 0],
        'col_1' :  [1, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        'col_2' :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
sProfile = sparse_profile(df, 'target')
print(sProfile.top_gain)
```
Output maximum gain obtained from each column
```
col_2    0.422810
col_1    0.074882
dtype: float64
```

```python
print(sProfile.report_sparsity)
```
Output percentage of zeros in column

```
col_1  0.8
col_2  0.1
```

Various sparse_profile reports can be accessed as attributes of the sparse_profile class object. List of all available attributes:
* report_sparsity:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pandas dataframe, Percentage of zeros in each column
* report_distinct:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, Count of distinct non zero values in each column
* report_overall:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, Overall summary of each column (similar to pandas describe())
* report_non_zero:&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, Summary of each column after removing zeros
* gain_df:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, Relative information gain at decile cutoffs for each column wrt target column
* auc_df:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, AUC of each column wrt target column
* top_gain:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pandas dataframe, Columns sorted by maximum gain obtained from gain_df