import librosa
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def stats2(arr):
    """Return [mean, std, min, max] — 4 stats per feature."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.zeros(4)
    return np.array([arr.mean(), arr.std(), arr.min(), arr.max()])

def extract_ml_features_from_array(y, sr):
    """
    Extracts fixed-length statistical features for classical ML from a raw audio array:
      - MFCCs + Δ + ΔΔ        → 78 features  (spectral timbre)
      - Prosodic (f0, RMS, ZCR) → 12 features  (rhythm & melody)
      - Formants F1, F2, F3    → 12 features  (vocal tract / vowel space)
    ─────────────────────────────────────────
    Total feature vector       → 102 features
    """
    try:
        # Force sample rate to 16000 if it isn't already
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        if len(y) == 0:
            return None

        frame_len = int(0.025 * sr)   # 25 ms
        hop_len   = int(0.010 * sr)   # 10 ms

        # ── A. MFCCs + Deltas ───────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                     n_fft=frame_len, hop_length=hop_len)
        if mfcc.shape[1] < 9:
            mfcc = np.pad(mfcc, ((0,0),(0, 9 - mfcc.shape[1])), mode='constant')

        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        combined    = np.vstack((mfcc, delta_mfcc, delta2_mfcc))  # (39, T)

        mfcc_mean = np.mean(combined, axis=1)   # 39
        mfcc_std  = np.std(combined,  axis=1)   # 39
        # subtotal: 78 features

        # ── B. Prosodic Features ───────────────────────────────────
        # RMS Energy — captures the "jagged vs smooth" energy envelope
        rms = librosa.feature.rms(
            y=y, frame_length=frame_len, hop_length=hop_len
        )[0]

        # f0 / Pitch contour — using yin for speed
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz
            fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
            sr=sr,
            hop_length=hop_len
        )

        # Create our own fast 'voiced_flag' using energy
        if np.max(rms) > 0:
            voiced_flag = rms > (0.05 * np.max(rms))
        else:
            voiced_flag = np.zeros_like(rms, dtype=bool)

        # Apply the flag
        min_len = min(len(f0), len(voiced_flag))
        f0_voiced = f0[:min_len][voiced_flag[:min_len]]
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

        if len(f0_voiced) == 0:
            f0_voiced = np.array([0.0])       # silent file fallback

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=frame_len, hop_length=hop_len
        )[0]

        # ── C. Formant Frequencies F1, F2, F3 ─────────────────────
        def estimate_formants_lpc(frame, sr, lpc_order=10, n_formants=3):
            frame = frame * np.hanning(len(frame))
            try:
                a      = librosa.lpc(frame, order=lpc_order)
                roots  = np.roots(a)
                roots  = roots[np.imag(roots) >= 0]
                angles = np.angle(roots)
                freqs  = sorted(angles * (sr / (2 * np.pi)))
                freqs  = [f for f in freqs if 90 < f < 4000]
                if len(freqs) < n_formants:
                    return np.zeros(n_formants)
                return np.array(freqs[:n_formants])
            except Exception:
                return np.zeros(n_formants)

        f1_list, f2_list, f3_list = [], [], []
        for start in range(0, len(y) - frame_len, hop_len):
            frame    = y[start : start + frame_len]
            formants = estimate_formants_lpc(frame, sr)
            f1_list.append(formants[0])
            f2_list.append(formants[1])
            f3_list.append(formants[2])

        f1 = np.array(f1_list)
        f2 = np.array(f2_list)
        f3 = np.array(f3_list)

        f1 = f1[f1 > 0]; f2 = f2[f2 > 0]; f3 = f3[f3 > 0]

        formant_features = np.concatenate([
            stats2(f1) if len(f1) > 0 else np.zeros(4),
            stats2(f2) if len(f2) > 0 else np.zeros(4),
            stats2(f3) if len(f3) > 0 else np.zeros(4),
        ])

        prosodic_features = np.concatenate([
            stats2(f0_voiced),
            stats2(rms),
            stats2(zcr),
        ])

        # ── Concatenate everything ─────────────────────────────────
        final_vector = np.concatenate([
            mfcc_mean,          # 39
            mfcc_std,           # 39
            prosodic_features,  # 12
            formant_features,   # 12
        ])
        return final_vector

    except Exception as e:
        print(f"Error processing audio array: {e}")
        return None

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def generate_inference_visualization_data(X_train, y_train_strings, X_test_sample, feature_names=None) -> dict:
    """
    Service function: Computes the 3D LDA space and SVM boundaries using TRAINING data,
    transforms a single TEST sample into this space, and extracts top LDA features.
    """
    # Attempt to extract feature names if X_train is a DataFrame and none were provided
    if feature_names is None and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()

    # 1. Scale the features based on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply the EXACT same scaling to your single test sample
    X_test_scaled = scaler.transform(X_test_sample)

    y_train_arr = np.array(y_train_strings)

    # 2. Fit Multi-Class LDA to get the optimal 3D space
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_train_lda_3d = lda.fit_transform(X_train_scaled, y_train_arr)
    # Transform the single test sample into this 3D space
    X_test_lda_3d = lda.transform(X_test_scaled)

    # --- NEW: Extract top 3 features for each LDA component ---
    lda_top_features = {}
    # lda.scalings_ has shape (n_features, n_components)
    for i in range(lda.scalings_.shape[1]):
        # Get the absolute weights to find the strongest magnitude contributors
        abs_weights = np.abs(lda.scalings_[:, i])
        # Get the indices of the top 3 weights (sorted ascending, so we take last 3 and reverse)
        top_3_indices = np.argsort(abs_weights)[-5:][::-1]
        
        component_features = []
        for idx in top_3_indices:
            name = feature_names[idx] if feature_names is not None else f"Feature_{idx}"
            # Store the name and the *original* signed weight to show direction
            component_features.append({
                "feature_name": name,
                "weight": float(lda.scalings_[idx, i])
            })
            
        lda_top_features[f"Component_{i+1}"] = component_features

    dialect_colors = {
        'Gulf Arabic': '#1f77b4',       
        'Maghrebi Arabic': '#d62728',   
        'Levantine Arabic': '#2ca02c',  
        'Egyptian Arabic': '#ff7f0e'    
    }

    # Calculate meshgrid boundaries based on the training data spread
    x_min, x_max = X_train_lda_3d[:, 0].min() - 1, X_train_lda_3d[:, 0].max() + 1
    y_min, y_max = X_train_lda_3d[:, 1].min() - 1, X_train_lda_3d[:, 1].max() + 1

    # Initialize the return dictionary with the new LDA top features
    viz_data = {
        "lda_top_features": lda_top_features,
        "dialects": {}
    }

    # 3. Compute SVMs and separate points for each dialect
    for dialect, color in dialect_colors.items():
        y_train_binary = (y_train_arr == dialect).astype(int)

        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_train_lda_3d, y_train_binary)
        
        w = svm.coef_[0]
        b = svm.intercept_[0]
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        zz = -(w[0] * xx + w[1] * yy + b) / (w[2] + 1e-10)

        target_mask = (y_train_binary == 1)
        other_mask = (y_train_binary == 0)

        viz_data["dialects"][dialect] = {
            "color": color,
            "train_target_points": {
                "x": X_train_lda_3d[target_mask, 0].tolist(),
                "y": X_train_lda_3d[target_mask, 1].tolist(),
                "z": X_train_lda_3d[target_mask, 2].tolist()
            },
            "train_other_points": {
                "x": X_train_lda_3d[other_mask, 0].tolist(),
                "y": X_train_lda_3d[other_mask, 1].tolist(),
                "z": X_train_lda_3d[other_mask, 2].tolist()
            },
            "hyperplane": {
                "x": xx.tolist(),
                "y": yy.tolist(),
                "z": zz.tolist()
            },
            "test_sample": {
                "x": X_test_lda_3d[:, 0].tolist(),
                "y": X_test_lda_3d[:, 1].tolist(),
                "z": X_test_lda_3d[:, 2].tolist()
            }
        }

    return viz_data


def predict_dialect(audio_path, trained_pipeline, label_encoder):
    """
    Loads an audio file, extracts features, and returns the predicted dialect.
    """
    print(f"Processing: {audio_path}")
    
    try:
        # 1. Load the audio file
        # Setting sr=16000 directly here saves time, but your extraction 
        # function will also double-check and handle it if needed.
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 2. Extract features using your custom function
        features = extract_ml_features_from_array(y, sr)
        
        if features is None:
            return "Error: Could not extract features from audio."
            
        # 3. Reshape features for the model (it expects a 2D array like [samples, features])
        features_2d = features.reshape(1, -1)
        
        # Get probabilities
        probabilities = trained_pipeline.predict_proba(features_2d)[0]
        
        # Find the highest probability
        max_prob_index = int(np.argmax(probabilities))
        confidence = float(probabilities[max_prob_index])
        
        # Convert index back to text
        predicted_label = label_encoder.inverse_transform([max_prob_index])[0]
        
        return predicted_label, round(confidence, 4), features

    except Exception as e:
        return f"Error processing file: {e}"