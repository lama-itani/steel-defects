# Computer vision to detect steel defetcs
Using computer vision (CV) to detect steel defects. Use case fit for automotive/manufacturing. Built for Cloudera.

## Project structure
```markdown
project-root/
├── train_job.py    # job script checks for new data, retrains and deploys based on criterion
├── data/
│   ├── raw/        # contains raw images, divided into training and validation along with their     annotation. This folder is included in the .gitignore
│   ├── interim/    # contains .csv with no duplicates and NaNs replaced by mean value
│   └── processed/
├── mlruns/         # for model logging (hidden and for local use only)
├── jobs/
│   ├── data_freshness_job.py         # checks periodically if there's new data
│   ├── model_training_job.py         # loads, preprocesses and trains/eval model
│   ├── model_deployment_job.py       # registers and deploys new model if perf criterion is fulfilled
├── notebooks/
│   ├── EDA_cv_steel.ipynb     # exploratory data analysis with visuals
│   ├── modular_cv_steel.ipynb # modular code
├── src/
│   ├── data_utils.py     # data loading & cleaning
│   ├── preprocessing.py  # preproc pipelines
│   └── models.py         # create and train models
│   └── mlflow_utils.py   # mlflow utilities w/ some MLOps best practices
```
## Next