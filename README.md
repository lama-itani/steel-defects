# Computer vision to detect steel defetcs
Using computer vision (CV) to detect steel defects. Use case fit for automotive/manufacturing. Built for Cloudera.

## Project structure (work-in-progress)
```markdown
project-root/
├── data/
│   ├── raw/        # contains raw images, divided into training and validation
│   └── processed/
├── mlruns/         # for model logging (hidden and for local use only)
├── jobs/           
│   ├── data_freshness_job.py         # checks periodically if there's new data
│   ├── model_training_job.py         # loads, preprocesses and trains/eval model
│   ├── model_deployment_job.py       # registers and deploys new model if perf criterion is fulfilled
├── notebooks/
│   ├── EDA_cv_steel.ipynb     # exploratory data analysis with visuals
│   ├── train_cv_steel.ipynb   # modular code in a notebook
├── src/
|   ├── utils/
│       ├── data_utils.py     # data loading & cleaning
|       ├── parse_xml.py      # parse xml to extract annotation for BBs
│       ├── preprocessing.py  # preproc pipelines
│       ├── models.py         # create and train models
│       └── mlflow_utils.py   # mlflow utilities w/ some MLOps best practices
```
## Next