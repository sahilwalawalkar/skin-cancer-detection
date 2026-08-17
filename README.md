# Skin Cancer Detection with Classical Machine Learning

**Author:** Sahil Walawalkar

This project explores binary classification of skin-lesion images as **benign** or **malignant** using principal component analysis (PCA) and classical machine-learning models. It contains a balanced image dataset, a Python analysis exported from a notebook, and two earlier R Markdown experiments with different train/test splits.

> **Important:** This is an educational data-mining project, not a medical device. Its predictions must not be used for diagnosis, treatment, or other clinical decisions.

## Project overview

The workflow converts RGB lesion images into flattened pixel vectors, reduces their dimensionality with PCA, and compares several classifiers:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM) with an RBF kernel
- Tuned Random Forest (Python analysis)

Model evaluation includes accuracy, confusion matrices, classification reports, ROC curves, ROC-AUC, five-fold cross-validation, and Random Forest hyperparameter tuning.

## Repository structure

```text
.
|-- benign/                              # 300 benign JPEG images
|-- malignant/                           # 300 malignant JPEG images
|-- skin_disease_detection.py            # Python analysis exported from Jupyter
|-- skin_cancer_50-50_finalproject.Rmd   # R analysis using a 50/50 split
|-- skin_cancer_80-20_finalproject.Rmd   # R analysis using an 80/20 split
`-- README.md
```

## Dataset

The repository contains 600 RGB JPEG images:

| Class | Label | Images |
|---|---:|---:|
| Benign | 0 | 300 |
| Malignant | 1 | 300 |
| **Total** |  | **600** |

All images are 224 × 224 pixels. A local integrity check found no unreadable files and no byte-identical duplicate images.

The original dataset source, license, lesion taxonomy, patient-level metadata, and collection protocol are not documented in the project files. Confirm that you have permission to redistribute and use these images before publishing or reusing the dataset. If the source becomes known, add its citation and license here.

## Python workflow

The Python analysis uses an 80/20 stratified split with `random_state=123` and follows this pipeline:

1. Load and label benign and malignant images.
2. Flatten each 224 × 224 × 3 image into 150,528 pixel features.
3. Normalize pixel values to the `[0, 1]` range.
4. Fit PCA on the training set and retain 50 components.
5. Train Logistic Regression, Random Forest, and SVM classifiers.
6. Evaluate held-out predictions and ROC-AUC.
7. Run stratified five-fold cross-validation.
8. Tune the Random Forest with `GridSearchCV`.

The results recorded in the script's conclusion are:

| Model | Test accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | Not recorded in the conclusion | Not recorded in the conclusion |
| Random Forest | 82.50% | 0.891 |
| SVM | 81.67% | 0.883 |
| Tuned Random Forest | 80.00% | Not recorded in the conclusion |

These are historical results embedded in the analysis, not independently reproduced during this repository review.

### Python setup

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib scikit-learn pillow jupyter
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn pillow jupyter
```

Before running the Python file:

1. Change `benign_dir` and `malignant_dir` to the local `benign/` and `malignant/` folders. The file currently contains machine-specific absolute paths.
2. Remove or comment out the raw `pip install ...` line near the top. It is notebook syntax exported into a `.py` file and causes a Python syntax error.
3. Run the analysis in an environment capable of displaying Matplotlib figures:

```bash
python skin_disease_detection.py
```

The full workflow can be memory- and compute-intensive because all image arrays and flattened feature matrices are held in memory, and the Random Forest grid search performs many fits.

## R Markdown workflows

The R analyses use packages including `jpeg`, `abind`, `caret`, `glmnet`, `imager`, `randomForest`, `e1071`, and `pROC`.

Install the dependencies in R:

```r
install.packages(c(
  "jpeg", "abind", "caret", "glmnet", "imager",
  "randomForest", "e1071", "pROC"
))
```

Open either `.Rmd` file in RStudio and update its dataset paths before running or knitting it. Both files currently reference machine-specific directories that are not part of this repository. The R analyses are retained as historical project work and may require code corrections before they run end to end; for example, some prediction inputs and SVM probability-column assumptions are inconsistent.

## Reproducibility notes

- The Python split and model configuration use seed `123`; the R analyses use seed `530`.
- PCA is fitted on training data in the Python workflow, which avoids using held-out test information during PCA fitting.
- Cross-validation is performed on already PCA-transformed training data. For stricter validation, put normalization and PCA inside a scikit-learn `Pipeline` so PCA is refitted within each fold.
- Images may originate from the same patient or lesion, but no grouping metadata is available. If such metadata exists, use group-aware splitting to prevent leakage.
- Accuracy alone is insufficient for clinical assessment. Sensitivity, specificity, calibration, subgroup performance, external validation, and clinically appropriate decision thresholds would also be necessary.

## Limitations and future work

- Replace hard-coded paths with paths relative to the repository.
- Restore the Python source as an `.ipynb` notebook or refactor it into a runnable script.
- Add a dependency lock file such as `requirements.txt` or `renv.lock`.
- Document the dataset source, license, class definitions, and patient-level provenance.
- Use a leakage-safe preprocessing/model pipeline.
- Evaluate convolutional neural networks or transfer learning with data augmentation.
- Add external validation and interpretable visual explanations.

## License

No software or dataset license is currently provided. Unless a license is added, others should not assume they have permission to copy, modify, or redistribute the code or images.

## Acknowledgments

The 80/20 R Markdown file identifies **Piyush Kolte** as its author. Add course, institution, collaborators, dataset creators, and formal dataset citations as appropriate.
