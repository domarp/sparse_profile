import pandas as pd
import numpy as np
from sparse_profile import sparse_profile

df = pd.DataFrame({
        'target' : [1, 1, 1, 1, 0, 0 ,0 ,0, 1, 0],
        'col_1' :  [1, 0, 0, 0, 0, 0, 0, 0, 0, 9],
        'col_2' :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
tt = sparse_profile(df, 'target', False)
print(tt.report_sparsity)

