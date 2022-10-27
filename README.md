# README for Pipeline

This document aims to explain how to use/set up and read the code for group 15.
The required packages can be found listed under 'Requirements' at the end of this document.

## Structure of folders

- app (contains all the code - you need to run main.py)
- data (contains processed CSV files and saved models)
- data_initial (contains only CSVs provided by the course)
- output (contains png and text files of the output)

## Structure of the code

The entire code was structured such that only running main.py is necessary.
Each step of the process was saved in functions created in various python files that main reads. 

main.py calls 10 functions from other python files.
The output of the 10 functions can be found in the output folder. While the scores of the models and the progression of the code can be seen in the program terminal.


## How to set up (Run main.py)
Make sure that the environment has the necessary packages specified in the section Requirments

There are 9 variables that you can change the values of. 
- All the validation variables refer to whether the cross-validation of the 4 models is run or skipped. We recommend running the code at least once without cross-validation. Running all the cross-validations can take up to 8 hours. In order to run the cross-validation of a model, you can change the value to 1
- All the model variables refer to whether you load the pre-saved models or run new ones. This task doesn't take long either way but it has been pre-saved to load the models. In order to fit the new models, you can change the value to 0
- The run_imp_crop variable refers to running the crop imputation. This has been set to 0 due to the fact that managing to install the necessary requirements will be extremely difficult. The package installation requires various package downgrades and upgrades and is very difficult.

The 6 names saved refer to the names of 4 CSVs and 2 text files. If you wish the output to be saved under different names then you can change these names.

## Reading the results
- All the figure and text files are saved in the output folder.
- The results of the 4 models will be visible in the terminal.
- The progress of the running of the code can be monitored in the terminal as well.


## Requirements
    python==3.8.13
    pandas==1.4.4
    numpy==1.23.1
    scikit-learn==1.1.2
    scipy==1.9.1
    matplotlib==3.5.2
    missingno==0.4.2
    seaborn==0.11.2
    itertools==0.1_3
    tqdm==4.64.1
    joblib==1.1.0

You require datawig==0.2.0 for the imputation of crop data. However, this has some dependencies that mess up the setup.

### Running the imputation of crop data
Since the setup for running the imputation of crop data is very hard, you can run it in Google Colab more easily. You can try such a run (although lengthy) at this link https://drive.google.com/drive/folders/1Yuq7v_lGaMhVcaR8zFtATXzL-afq1afU
