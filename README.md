# Sprint 3
* Made summary of missing values per column
* Imputed `MAM` and `total_population`
* Used `GAM - SAM` to predict `MAM`
* **NOT REQUIRED** [Used LOCF for `total_population`. Only valuess from 2021-07-01 were missing, so I imputed the previous known population value for each district]
* kNN for columns: `ndvi`, `ipc`, `total_conflicts`, `price of water`
* MICE for `price of water`
* Spline imputation + NOCB(next observation carried backward) for `total_conflicts`


