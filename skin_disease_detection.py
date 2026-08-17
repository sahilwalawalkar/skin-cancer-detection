#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib scikit-learn pillow


# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score

print("Libraries imported successfully!")


# In[7]:


from pathlib import Path
from PIL import Image
import numpy as np

# Folder paths
benign_dir = Path(
    r"F:\MastersHW\Sem1\Advance Data Mining\Project\benign"
)

malignant_dir = Path(
    r"F:\MastersHW\Sem1\Advance Data Mining\Project\malignant"
)


def load_images_from_folder(folder, label):
    images = []
    labels = []

    for file_path in folder.iterdir():
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                img = Image.open(file_path).convert("RGB")
                img_array = np.array(img)

                images.append(img_array)
                labels.append(label)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return images, labels


# Load benign images -> 0
benign_images, benign_labels = load_images_from_folder(
    benign_dir, 0
)

# Load malignant images -> 1
malignant_images, malignant_labels = load_images_from_folder(
    malignant_dir, 1
)


print("Benign images:", len(benign_images))
print("Malignant images:", len(malignant_images))
print(
    "Total images:",
    len(benign_images) + len(malignant_images)
)


# In[8]:


print("First benign image shape:", benign_images[0].shape)
print("First malignant image shape:", malignant_images[0].shape)

print("Benign label:", benign_labels[0])
print("Malignant label:", malignant_labels[0])


# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(benign_images[0])
axes[0].set_title("Benign - Label 0")
axes[0].axis("off")

axes[1].imshow(malignant_images[0])
axes[1].set_title("Malignant - Label 1")
axes[1].axis("off")

plt.show()


# In[10]:


# Combine benign and malignant images
X = np.array(benign_images + malignant_images)
y = np.array(benign_labels + malignant_labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Check class distribution
print("Benign samples:", np.sum(y == 0))
print("Malignant samples:", np.sum(y == 1))


# In[11]:


# Split dataset: 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=123,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)


# In[12]:


# Flatten images

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("Original training shape:", X_train.shape)
print("Flattened training shape:", X_train_flat.shape)

print("Original testing shape:", X_test.shape)
print("Flattened testing shape:", X_test_flat.shape)


# In[13]:


# Convert pixel values to float and normalize from 0-255 to 0-1

X_train_flat = X_train_flat.astype("float32") / 255.0
X_test_flat = X_test_flat.astype("float32") / 255.0

print("Training data range:",
      X_train_flat.min(),
      "to",
      X_train_flat.max())

print("Testing data range:",
      X_test_flat.min(),
      "to",
      X_test_flat.max())


# In[14]:


# Apply PCA to reduce 150,528 features -> 50 components

pca = PCA(n_components=50, random_state=123)

X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

print("PCA training shape:", X_train_pca.shape)
print("PCA testing shape:", X_test_pca.shape)

print("Total explained variance:",
      pca.explained_variance_ratio_.sum())


# In[15]:


plt.figure(figsize=(8, 6))

plt.scatter(
    X_test_pca[y_test == 0, 0],
    X_test_pca[y_test == 0, 1],
    label="Benign (0)",
    alpha=0.6
)

plt.scatter(
    X_test_pca[y_test == 1, 0],
    X_test_pca[y_test == 1, 1],
    label="Malignant (1)",
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Plot of Test Data")
plt.legend()

plt.show()


# In[16]:


# Train Logistic Regression model

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=123
)

logistic_model.fit(X_train_pca, y_train)

print("Logistic Regression model trained successfully!")


# In[17]:


# Make predictions

logistic_predictions = logistic_model.predict(X_test_pca)

# Calculate accuracy
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)


# In[18]:


# Confusion Matrix

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

print("Confusion Matrix:")
print(logistic_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Benign", "Malignant"]
    )
)


# In[19]:


# Train Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_features=7,
    random_state=123,
    n_jobs=-1
)

rf_model.fit(X_train_pca, y_train)

print("Random Forest model trained successfully!")


# In[20]:


# Make predictions

rf_predictions = rf_model.predict(X_test_pca)

# Calculate accuracy
rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)


# In[21]:


# Random Forest Confusion Matrix

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

print("Random Forest Confusion Matrix:")
print(rf_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Benign", "Malignant"]
    )
)


# In[22]:


# Probability predictions for malignant class (1)

rf_probabilities = rf_model.predict_proba(X_test_pca)[:, 1]

# Calculate AUC
rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print(
    "Random Forest AUC:",
    round(rf_auc, 3)
)


# In[23]:


# Calculate ROC curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probabilities
)

# Plot ROC curve

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {rf_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# In[24]:


# Train SVM model

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=123
)

svm_model.fit(X_train_pca, y_train)

print("SVM model trained successfully!")


# In[25]:


# Make SVM predictions

svm_predictions = svm_model.predict(X_test_pca)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print(
    "SVM Accuracy:",
    round(svm_accuracy * 100, 2),
    "%"
)


# In[26]:


# SVM Confusion Matrix

svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)

print("SVM Confusion Matrix:")
print(svm_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=["Benign", "Malignant"]
    )
)


# In[27]:


# Get SVM probabilities for malignant class (1)

svm_probabilities = svm_model.predict_proba(X_test_pca)[:, 1]

# Calculate AUC
svm_auc = roc_auc_score(
    y_test,
    svm_probabilities
)

print(
    "SVM AUC:",
    round(svm_auc, 3)
)


# In[28]:


# Calculate SVM ROC curve

svm_fpr, svm_tpr, svm_thresholds = roc_curve(
    y_test,
    svm_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"SVM (AUC = {svm_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# In[29]:


# Logistic Regression probabilities for malignant class (1)

logistic_probabilities = logistic_model.predict_proba(
    X_test_pca
)[:, 1]

# Calculate AUC
logistic_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)

print(
    "Logistic Regression AUC:",
    round(logistic_auc, 3)
)


# In[30]:


# ROC values for Logistic Regression

log_fpr, log_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

# ROC values for Random Forest

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

# ROC values for SVM

svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_probabilities
)


# Plot all ROC curves

plt.figure(figsize=(9, 7))

plt.plot(
    log_fpr,
    log_tpr,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"SVM (AUC = {svm_auc:.3f})"
)

# Random classifier baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison - Skin Disease Detection"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# In[31]:


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ],
    
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy,
        svm_accuracy
    ],
    
    "ROC-AUC": [
        logistic_auc,
        rf_auc,
        svm_auc
    ]
})

results["Accuracy"] = (
    results["Accuracy"] * 100
).round(2)

results["ROC-AUC"] = (
    results["ROC-AUC"]
).round(3)

results


# In[32]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

print("Cross-validation configured successfully!")


# In[33]:


logistic_cv_scores = cross_val_score(
    logistic_model,
    X_train_pca,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Logistic Regression CV Scores:")
print(logistic_cv_scores)

print(
    "Mean CV Accuracy:",
    round(logistic_cv_scores.mean() * 100, 2),
    "%"
)

print(
    "Standard Deviation:",
    round(logistic_cv_scores.std() * 100, 2),
    "%"
)


# In[34]:


rf_cv_scores = cross_val_score(
    rf_model,
    X_train_pca,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Random Forest CV Scores:")
print(rf_cv_scores)

print(
    "Mean CV Accuracy:",
    round(rf_cv_scores.mean() * 100, 2),
    "%"
)

print(
    "Standard Deviation:",
    round(rf_cv_scores.std() * 100, 2),
    "%"
)


# In[35]:


svm_cv_scores = cross_val_score(
    svm_model,
    X_train_pca,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("SVM CV Scores:")
print(svm_cv_scores)

print(
    "Mean CV Accuracy:",
    round(svm_cv_scores.mean() * 100, 2),
    "%"
)

print(
    "Standard Deviation:",
    round(svm_cv_scores.std() * 100, 2),
    "%"
)


# In[36]:


final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM"
    ],

    "Test Accuracy (%)": [
        logistic_accuracy * 100,
        rf_accuracy * 100,
        svm_accuracy * 100
    ],

    "ROC-AUC": [
        logistic_auc,
        rf_auc,
        svm_auc
    ],

    "CV Mean Accuracy (%)": [
        logistic_cv_scores.mean() * 100,
        rf_cv_scores.mean() * 100,
        svm_cv_scores.mean() * 100
    ],

    "CV Std (%)": [
        logistic_cv_scores.std() * 100,
        rf_cv_scores.std() * 100,
        svm_cv_scores.std() * 100
    ]
})

final_results = final_results.round(3)

final_results


# In[37]:


from sklearn.model_selection import GridSearchCV

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["sqrt", 7, 10],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=123,
        n_jobs=-1
    ),
    param_grid=rf_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_pca, y_train)


# In[38]:


print("Best Parameters:")
print(rf_grid.best_params_)

print(
    "\nBest Cross-Validation Accuracy:",
    round(rf_grid.best_score_ * 100, 2),
    "%"
)


# In[40]:


best_rf_model = rf_grid.best_estimator_

print(best_rf_model)


# In[41]:


tuned_rf_predictions = best_rf_model.predict(X_test_pca)

tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_predictions
)

print(
    "Tuned Random Forest Test Accuracy:",
    round(tuned_rf_accuracy * 100, 2),
    "%"
)


# In[42]:


tuned_rf_cm = confusion_matrix(
    y_test,
    tuned_rf_predictions
)

print("Tuned Random Forest Confusion Matrix:")
print(tuned_rf_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tuned_rf_predictions,
        target_names=["Benign", "Malignant"]
    )
)


# In[43]:


tuned_rf_probabilities = best_rf_model.predict_proba(
    X_test_pca
)[:, 1]

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_probabilities
)

print(
    "Tuned Random Forest AUC:",
    round(tuned_rf_auc, 3)
)


# In[44]:


# Final model results

model_names = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Tuned Random Forest"
]

accuracies = [
    logistic_accuracy * 100,
    rf_accuracy * 100,
    svm_accuracy * 100,
    tuned_rf_accuracy * 100
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    model_names,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Machine Learning Model")
plt.title("Model Accuracy Comparison")

plt.ylim(0, 100)

# Add accuracy values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ## Conclusion
# 
# This project developed a machine learning pipeline for classifying skin lesion images as benign or malignant.
# 
# The original image dataset contained 600 images, with 300 benign and 300 malignant samples. Images were resized to 224 × 224 × 3 and flattened into 150,528 pixel features. Principal Component Analysis (PCA) was used to reduce the dimensionality to 50 principal components before classification.
# 
# Three machine learning algorithms were evaluated:
# 
# - Logistic Regression
# - Random Forest
# - Support Vector Machine (SVM)
# 
# Random Forest achieved the strongest held-out test performance with an accuracy of 82.50% and ROC-AUC of 0.891. SVM achieved a comparable accuracy of 81.67% and ROC-AUC of 0.883.
# 
# Five-fold cross-validation showed that both Random Forest and SVM produced relatively stable performance across different subsets of the training data.
# 
# Hyperparameter tuning was also performed using GridSearchCV. Although the tuned Random Forest achieved higher cross-validation accuracy, its final test accuracy was 80.00%, which did not outperform the original Random Forest configuration.
# 
# Overall, the results demonstrate that classical machine learning techniques combined with PCA can effectively distinguish between benign and malignant skin lesion images. Future improvements could include convolutional neural networks, transfer learning, data augmentation, and larger image datasets.

# In[ ]:




