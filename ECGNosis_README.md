# ECGnosis

ECGNosis is an ECG signal classification project developed using Python and deep learning. It uses the InceptionTime model to analyse ECG recordings and classify them based on the AF label in the dataset.

The project covers ECG signal preprocessing, model training, performance evaluation and prediction generation. It also separates the prediction results into high-risk and low-risk categories for further review.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- h5py
- Scikit-learn
- Matplotlib
- sktime (InceptionTime)

## Project Files

| File | Description |
|---|---|
| `ecg_pipeline_inceptiontime.py` | Preprocesses ECG signals, trains the model and evaluates its performance. |
| `risk_seperator.py` | Separates prediction results into high-risk and low-risk files. |
| `look.py` | Displays the datasets and structure stored in an HDF5 file. |
| `confusion_matrix.png` | Confusion matrix generated during evaluation. |
| `roc_curve.png` | ROC curve generated during evaluation. |
| `performance_metrics.csv` | Stores the model evaluation metrics. |
| `ecg_risk_predictions.csv` | Contains the prediction results for each exam. |
| `high_risk_sorted.csv` | Contains high-risk predictions sorted by prediction probability. |
| `low_risk_sorted.csv` | Contains low-risk predictions sorted by prediction probability. |

## Dataset

The main script expects the following files:

- `exams_part17.hdf5` – ECG recordings used for training.
- `exams_part16.hdf5` – ECG recordings used for testing.
- `exams.csv` – Exam details and labels.

The HDF5 files should contain `tracings` and `exam_id` datasets. The CSV file should contain the `exam_id` and `AF` columns.

The dataset files are not included in this repository. They need to be obtained separately from an authorized source.

## How the Project Works

1. ECG recordings and exam IDs are loaded from the HDF5 files.
2. The exam IDs are matched with the labels in the CSV file.
3. A band-pass filter is applied to reduce noise in the ECG signals.
4. The signals are resampled from 500 Hz to 250 Hz and normalized.
5. The InceptionTime model is trained using the training data.
6. The model predicts labels for the test data.
7. The predictions are evaluated and saved as CSV files and plots.

## Installation

Use Python 3 and install the required packages:

```bash
pip install numpy pandas scipy h5py scikit-learn matplotlib sktime
```

The InceptionTime classifier may also require additional deep-learning dependencies, depending on the installed version of sktime. Install the compatible dependencies if they are not already available.

## Running the Project

First, place the dataset files in a folder on your computer.

Open `ecg_pipeline_inceptiontime.py` and update the `BASE_DIR` value near the bottom of the file with the location of your dataset:

```python
BASE_DIR = r"/path/to/your/dataset"
```

Then run the script:

```bash
python ecg_pipeline_inceptiontime.py
```

After execution, the script creates a `results` folder in the dataset directory. It saves the confusion matrix, ROC curve, performance metrics and ECG prediction results there.

To separate the predictions into high-risk and low-risk files, run:

```bash
python risk_seperator.py
```

Run this command from the project directory after the prediction CSV has been generated.

## Model Results

The current `performance_metrics.csv` file contains these results:

| Metric | Value |
|---|---:|
| Accuracy | 0.97695 |
| Precision | 0.19231 |
| Recall | 0.01124 |
| F1-score | 0.02123 |
| ROC-AUC | 0.58129 |

These values are from the results currently stored in the repository. Accuracy is high, but recall and ROC-AUC are low. This means the model misses many positive cases at the current classification threshold. The results need further improvement and validation.

## Note

The current code uses the `AF` label from the dataset. It should not be considered a Chagas disease detection model unless a Chagas-specific dataset, labels and evaluation process are implemented.

This project is for academic and learning purposes. It is not intended for clinical diagnosis or medical decision-making.

## Author

Kevin Joel B.
