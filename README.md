# classification-ml-pipeline

An end-to-end machine learning pipeline that classifies the cell-type of seedling crop plants. This pipeline includes:
- Taking in training and testing data from the user as a command-line argument
- Preprocessing training and testing embeddings and meta data
- Performing hyperparameter tuning and cross validation on the specified model
- Training the model on training dataset
- Evaluating performance of model on independent testing data with confusion matrix, PRC, and ROC
- Predicting the cell-type from unknown seedling data

Future features: 
- Model deployment on the cloud
- CI/CD pipeline
- Monitoring and triggering
- Automated model retraining