| Model | Key Hyperparameters | Search Method |
| Ridge | alpha ∈ {0.01, 0.1, 1, 10, 100, 1000} | RidgeCV (3-fold) |
| Lasso | alpha (automatic) | LassoCV (3-fold) |
| Random Forest | n_estimators=180, max_depth=14, min_samples_leaf=3 | Fixed (pilot-tuned) |
| Gradient Boosting | The sklearn defaults (n_estimators=100, learning_rate=0.1, max_depth=3) are as follows: | Fixed |
| HistGradientBoosting | learning_rate=0.055,max_iter=240,l2_regularization=0.05 | Fixed (pilot-tuned) |
| XGBoost | n_estimators=220, learning_rate=0.045, max_depth=4, subsample=colsample=0.85 | Fixed (pilot-tuned) |
| LightGBM | n_estimators=260, learning_rate=0.045, num_leaves=31, subsample=colsample=0.85 | Fixed (pilot-tuned) |
| CatBoost | iterations=260, learning_rate=0.045, depth=6, l2_leaf_reg=3 | Fixed (pilot-tuned) |
| Ridge | alpha ∈ {0.01, 0.1, 1, 10, 100, 1000} | RidgeCV (3-fold internal) |
| Lasso | alpha (automatic path) | LassoCV (3-fold internal) |
| Tree-based learners (RF/GBDT/HistGB/XGBoost/LightGBM/CatBoost) | Same as the tree-based rows above | Fixed after pilot tuning |
| StackingRidge meta-learner | Ridge alpha = 1.0 | Default (sensitivity - verified) |
