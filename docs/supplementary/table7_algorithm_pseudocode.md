| Step | Operation |
| Input | Training set ,  base learners , meta-learner , CV folds |
| Stage 1: Create meta - features: 1 | Partition training set into  folds: |
| 2 | For  do |
| 3 | Training set ,  validation set |
| 4 | For  do |
| 5 | Train  on , make predictions for samples in |
| 6 | End for |
| 7 | End for |
| 8 | Create meta-feature matrix |
| Stage 2: Fit meta-learner: 9 | Use  as a training data, fit Ridge meta-learner: |
| Stage 3: Make final predictions: 10 | Retrain all base learners  on complete training data |
| 11 | For new sample , calculate meta-features |
| 12 | Final prediction |
| Output | Meta-learner coefficients , weights of the meta-learners, prediction value |
| OOF-TSRM two-stage residual model | OOF-TSRM two-stage residual model |
| Input | Training set ,  candidate global model pool , residual model , CV folds |
| Stage 1: Global model selection and OOF prediction: 1 | Partition training set into  folds: |
| 2 | Per candidate model  do |
| 3 | For  do |
| 4 | Train  on , produce OOF forecasts of |
| 5 | End for |
| 6 | Compute CV MAE for model |
| 7 | End for |
| 8 | Select model with smallest CV MAE as global model |
| 9 | Generate OOF predictions of global model |
| Stage 2: Residual correction: 10 | Compute OOF residuals: |
| 11 | Use  as training data, fit residual correction model |
| Stage 3: Final prediction: 12 | Retrain global model  on complete training set |
| 13 | For new sample , final prediction |
| Output | Global model , residual correction model , final prediction value |
