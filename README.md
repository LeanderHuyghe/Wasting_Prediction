# Sprint 2
### Step 1: Integrating the baseline models with the preprocessed datasets
`This section will be updated based on the progress of the project`
### Progress
1. Created .py file for our data prepartion functions 
2. Compared our datasets
3. Implemented baseline model with semiyearly and monthly data (**best solution yet to be found**)

### Problems
RF cannot work with `NaN` values. Therefore, our problems to solve:
1. If we decide to drop rows with nan values, the dataframe becomes empty
2. If we remove only the columns which have nan values, we lose our target variable `next_prevalence` 

A janky solution has been implemented within the files `Baseline Model on Monthly Preprocessed Data.ipynb` and `Baseline Model on Semiyearly Preprocessed Data.ipynb`.