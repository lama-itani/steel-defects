# Predicting sedimentation velocity using ML  
ML project to predict sedimentation velocity. Contains some MLOps best practices.
## Project structure
```markdown
project-root/
├── train_job.py    # job script checks for new data, retrains and deploys based on criterion
├── data/
│   ├── raw/        # contains raw .csv file
│   ├── interim/    # contains .csv with no duplicates and NaNs replaced by mean value
│   └── processed/
├── mlruns/         # for model logging (hidden and for local use only)
├── jobs/
│   ├── data_freshness_job.py         # checks periodically if there's new data
│   ├── model_training_job.py         # loads, preprocesses and trains/eval model
│   ├── model_deployment_job.py       # registers and deploys new model if perf criterion is fulfilled
├── notebooks/
│   ├── EDA_ml_sediment.ipynb     # exploratory data analysis with visuals
│   ├── modular_ml_sediment.ipynb # modular code
├── src/
│   ├── data_utils.py     # data loading & cleaning
│   ├── preprocessing.py  # preproc pipelines
│   └── models.py         # create and train models
│   └── mlflow_utils.py   # mlflow utilities w/ some MLOps best practices
```
## Next