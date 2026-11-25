import os
import numpy as np
import pandas as pd
import h5py
import gc
from scipy.signal import butter, sosfiltfilt, resample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from sktime.classification.deep_learning import InceptionTimeClassifier


def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def denoise_signal(signal, lowcut=0.5, highcut=40.0, fs=500):
    sos = butter_bandpass_sos(lowcut, highcut, fs)
    return sosfiltfilt(sos, signal)


def preprocess_ecg_batch(batch, target_fs=250, original_fs=500):
    processed_batch = []
    
    for sig in batch:
        try:
            clean_channels = []
            for i in range(sig.shape[1]):
                clean_channel = denoise_signal(sig[:, i], fs=original_fs)
                clean_channels.append(clean_channel)
            
            
            clean_sig = np.column_stack(clean_channels)
            
            
            new_length = int(len(clean_sig) * target_fs / original_fs)
            resampled = resample(clean_sig, new_length, axis=0)
            
            
            mean = np.mean(resampled, axis=0, keepdims=True)
            std = np.std(resampled, axis=0, keepdims=True)
            resampled = (resampled - mean) / (std + 1e-8)
            
            processed_batch.append(resampled)
            
            
            del clean_sig, clean_channels
            
        except Exception as e:
            print(f"[Warning] Preprocessing failed for sample: {e}")
            
            processed_batch.append(np.zeros((int(4096 * target_fs / original_fs), sig.shape[1]), dtype=np.float32))
    
    return np.array(processed_batch, dtype=np.float32)


def preprocess_ecg_signals(X, target_fs=250, batch_size=100):
    n_samples = len(X)
    processed = []
    
    print(f"Processing {n_samples} signals in batches of {batch_size}...")
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        print(f"  Batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1} "
              f"(samples {i}-{batch_end})", end='\r')
        
        batch = X[i:batch_end]
        processed_batch = preprocess_ecg_batch(batch, target_fs)
        processed.append(processed_batch)
        
        
        del batch, processed_batch
        gc.collect()
    
    print() 
    
    
    result = np.concatenate(processed, axis=0)
    del processed
    gc.collect()
    
    return result



def run_ecg_inception_pipeline(base_dir):
    """Full ECG deep learning classification pipeline."""

    print("ECG Classification Pipeline Started")
    
    train_h5 = os.path.join(base_dir, "exams_part17.hdf5") 
    test_h5 = os.path.join(base_dir, "exams_part16.hdf5")   
    csv_file = os.path.join(base_dir, "exams.csv")
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    
    for file_path in [train_h5, test_h5, csv_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    print("All input files found.")

    
    print("\nLoading Training Dataset...")
    with h5py.File(train_h5, "r") as f:
        X_train_raw = np.array(f["tracings"])
        exam_id_train = np.array(f["exam_id"])

    print(f"Training data loaded: {X_train_raw.shape}")
    print(f"Training data size: {X_train_raw.nbytes / 1e9:.2f} GB")

    
    print("\nLoading Testing Dataset...")
    with h5py.File(test_h5, "r") as f:
        X_test_raw = np.array(f["tracings"])
        exam_id_test = np.array(f["exam_id"])

    print(f"Testing data loaded: {X_test_raw.shape}")
    print(f"Testing data size: {X_test_raw.nbytes / 1e9:.2f} GB")

    
    print("\nLoading Labels CSV...")
    df = pd.read_csv(csv_file)
    if "exam_id" not in df.columns:
        raise KeyError("'exam_id' column not found in CSV!")

    print(f"CSV Columns: {df.columns.tolist()}")

    df_train = pd.DataFrame({"exam_id": exam_id_train}).merge(df, on="exam_id", how="inner")
    df_test = pd.DataFrame({"exam_id": exam_id_test}).merge(df, on="exam_id", how="inner")

    
    if "AF" not in df_train.columns:
        raise KeyError("'AF' label column not found in CSV!")

    y_train = df_train["AF"].astype(int).values
    y_test = df_test["AF"].astype(int).values
    
    
    train_valid_indices = df_train.index.tolist()
    test_valid_indices = df_test.index.tolist()

    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    
    print("\nPreprocessing ECG Signals (denoise + resample + normalize)...")
    print("Using memory-optimized batch processing...")
    
    
    BATCH_SIZE = 50  
    
    X_train_filtered = X_train_raw[train_valid_indices]
    X_test_filtered = X_test_raw[test_valid_indices]
    
    print(f"Filtered X_train: {X_train_filtered.shape}, y_train: {y_train.shape}")
    print(f"Filtered X_test: {X_test_filtered.shape}, y_test: {y_test.shape}")
    
    X_train = preprocess_ecg_signals(X_train_filtered, batch_size=BATCH_SIZE)
    del X_train_raw, X_train_filtered  
    gc.collect()
    
    X_test = preprocess_ecg_signals(X_test_filtered, batch_size=BATCH_SIZE)
    del X_test_raw, X_test_filtered  
    gc.collect()
    
    print(f"Preprocessed shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    
    print("\nBuilding InceptionTime Model...")
    model = InceptionTimeClassifier(
        n_epochs=15,
        batch_size=32,
        verbose=True,
        random_state=42,
    )

    
    print("\nTraining the Model...")
    print("Note: This may take a while depending on your hardware...")
    model.fit(X_train, y_train)
    print(" Model training completed!")
    
    
    del X_train
    gc.collect()

    
    print("\n Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n Performance Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred, zero_division=0))

    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"]).plot(
        cmap="Blues", values_format="d", ax=ax
    )
    plt.title("Confusion Matrix - ECG Classification")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f" Confusion matrix saved to: {cm_path}")
    plt.close()

    y_test_bin = label_binarize(y_test, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - InceptionTime Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f" ROC curve saved to: {roc_path}")
    plt.close()

    print("\nClassifying Patients as High Risk / Low Risk...")
    risk_labels = np.where(y_pred == 1, "High Risk", "Low Risk")
    results_df = pd.DataFrame({
        "Exam_ID": df_test["exam_id"].values,
        "True_Label": y_test,
        "Predicted_Label": y_pred,
        "Prediction_Probability": y_pred_prob,
        "Risk_Category": risk_labels
    })

    output_file = os.path.join(output_dir, "ecg_risk_predictions.csv")
    results_df.to_csv(output_file, index=False)

    print(f"\n Results saved to: {output_file}")
    
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Value": [acc, prec, rec, f1, roc_auc]
    })
    metrics_file = os.path.join(output_dir, "performance_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f" Metrics saved to: {metrics_file}")
    
    print("\n Pipeline executed successfully.")

    return model, results_df



if __name__ == "__main__":
    BASE_DIR = r"C:\Users\manoj\OneDrive\Desktop\CDPS"
    
    try:
        model, results = run_ecg_inception_pipeline(BASE_DIR)
    except Exception as e:
        print(f"\n Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()