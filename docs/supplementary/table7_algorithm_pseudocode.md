| Step | Operation |
| Input | Training set D={xi,yi}i=1n, K base learners {Fk}k=1K, meta-learner G, CV folds V |
| Stage 1: Create meta - features: 1 | Partition training set into V folds: D=⋃v=1VDv |
| 2 | For v=1,2,…,V do |
| 3 | Training set Dtrainv=D∖Dv, validation set Dvalv=Dv |
| 4 | For k=1,2,…,K do |
| 5 | Train Fkv on Dtrainv, make predictions for samples in Dvalv zi,k=Fkvxi |
| 6 | End for |
| 7 | End for |
| 8 | Create meta-feature matrix Z=zi,k∈ℝn×K |
| Stage 2: Fit meta-learner: 9 | Use Z,y as a training data, fit Ridge meta-learner: w=Z⊤Z+αI−1Z⊤y |
| Stage 3: Make final predictions: 10 | Retrain all base learners {Fkfull}k=1K on complete training data D |
| 11 | For new sample x∗, calculate meta-features z∗=Fkfullx∗k=1K |
| 12 | Final prediction y∗=k=1Kwk⋅z∗,k |
| Output | Meta-learner coefficients w, weights of the meta-learners, prediction value y∗ |
| OOF-TSRM two-stage residual model |
| Input | Training set D={xi,yi}i=1n, candidate global model pool ℳ, residual model F2, CV folds V |
| Stage 1: Global model selection and OOF prediction: 1 | Partition training set into V folds: D=⋃v=1VDv |
| 2 | Per candidate model M∈ℳ do |
| 3 | For v=1,2,…,V do |
| 4 | Train Mv on D∖Dv, produce OOF forecasts of Dv |
| 5 | End for |
| 6 | Compute CV MAE for model M |
| 7 | End for |
| 8 | Select model with smallest CV MAE as global model F1 |
| 9 | Generate OOF predictions of global model {yi1}i=1n |
| Stage 2: Residual correction: 10 | Compute OOF residuals: εi=yi−yi1 |
| 11 | Use {xi,εi} as training data, fit residual correction model F2 |
| Stage 3: Final prediction: 12 | Retrain global model F1full on complete training set |
| 13 | For new sample x∗, final prediction y∗=F1fullx∗+F2x∗ |
| Output | Global model F1full, residual correction model F2, final prediction value y∗ |
