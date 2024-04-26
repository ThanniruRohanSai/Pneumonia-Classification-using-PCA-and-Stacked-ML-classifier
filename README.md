# Pneumonia-Classification-using-PCA-and-Stacked-ML-classifier

This project investigates the use of Principal Component Analysis (PCA) for dimensionality reduction and stacked machine learning classifiers to achieve high accuracy in pneumonia classification using chest X-ray images.

Data:

The project utilizes the Chest X-ray Pneumonia dataset publicly available on Kaggle. You can access the dataset here: https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy

The dataset consists of chest X-ray images labelled as pneumonia or normal.

Methodology:

Data Preprocessing:
Chest X-ray images are preprocessed to ensure consistency (e.g., resizing, normalization).
Labels are encoded into numerical representations for machine learning algorithms.
Principal Component Analysis (PCA):
PCA is employed to reduce the dimensionality of the image data. This can be beneficial for:
Reducing computational complexity during model training.
Potentially mitigating the "curse of dimensionality" and improving model performance.
Stacked Machine Learning Classifiers:
A stacked classifier approach is implemented, achieving an accuracy of 96% on the test set.
Four base classifiers are used: Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and XGBoost.
Each base classifier is trained on the PCA-reduced features to identify patterns indicative of pneumonia.
The predictions from the base classifiers are then used as input features for a final Random Forest classifier. This final classifier aims to learn from the combined insights of the base models, potentially leading to improved overall accuracy.

Evaluation:

The stacked classifier achieved a remarkable accuracy of 96% on the held-out test set. This indicates that the model can effectively generalize its learning to unseen pneumonia cases with high success. We will further evaluate the model's performance using additional metrics like precision, recall, and F1-score to provide a more comprehensive analysis.

Benefits of Stacked Classifiers:

Ensemble methods like stacked classifiers can leverage the strengths of different base models, potentially leading to more robust and accurate predictions compared to a single classifier.
By combining the knowledge from multiple models, the stacked classifier can potentially capture more complex relationships within the data.

Future Work:

Explore the use of alternative dimensionality reduction techniques like Autoencoders or t-SNE.
Investigate the impact of hyperparameter tuning on the performance of both base classifiers and the final stacked classifier.
Experiment with different stacking architectures and final ensemble models to identify the most effective configuration for pneumonia classification.
Visualize the features learned by the PCA and the stacked classifier to gain insights into the underlying patterns used for pneumonia detection.
