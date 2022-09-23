# Sprint 2
### Step 1: Integrating the baseline models with the preprocessed datasets
`This section will be updated based on the progress of the project`

### Data configuration
* X = all features which are not catgeorical or object. Removed `increase`, `prevalence`, and `next_prevalence` 
* y = `next_prevalence`
### Progress
1. Created .py file for our data prepartion functions 
2. Compared our datasets
3. Implemented baseline model with semiyearly and monthly data (**best solution yet to be found**)

Convert `.ipynb` to `.py` easily:
```
jupyter nbconvert --to script "notebook.ipynb"
```

### Problems
RF cannot work with `NaN` values. Therefore, some issues to solve:
1. If we decide to drop rows with nan values, the dataframe becomes empty
2. If we remove only the columns which have nan values, we lose our target variable `next_prevalence` 



