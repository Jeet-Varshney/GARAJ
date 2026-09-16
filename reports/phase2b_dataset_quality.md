# GARAJ Phase 2B — Dataset Quality & Baseline Evaluation Report

## Production Model Status Notice

> [!IMPORTANT]
> - **Production Model**: `LA_model.pth`
> - **Production Accuracy Claim**: **NOT ESTABLISHED** (Evaluating on newly ingested Real + Synthetic dataset)
> - **Controlled Synthetic-Only Result**: `94.44% (17/18)` — *Synthetic-only controlled signal accuracy — NOT overall model accuracy.*

---

## 1. Dataset Quality Summary

- **Total Ingested Files**: `65`
- **Verified Real Human Samples**: `21`
- **Verified Synthetic Samples**: `44`
- **Language Distribution**:
  - **Hindi**: `13`
  - **Hinglish**: `14`
  - **English**: `38`
- **Unique Speaker Count**: `15`
- **Unique Generator Count**: `3` (`Google gTTS, Microsoft EdgeTTS, genuine_human`)
- **Device Count**: `3` (`browser_microphone, headset_mic, synthetic_stream`)
- **Duplicate Audio Files**: `0`
- **Invalid / Unreadable Files**: `0`
- **Missing Required Metadata**: `0`

---

## 2. Speaker-Disjoint Split Statistics

| Split | Total Samples | Real (Bona-Fide) | Synthetic (Spoof) | Unique Speakers | Split File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `39` | `5` | `34` | `9` | `datasets/splits/train.json` |
| **Validation** | `15` | `10` | `5` | `3` | `datasets/splits/validation.json` |
| **Test** | `11` | `6` | `5` | `3` | `datasets/splits/test.json` |

**Speaker Disjoint Check**: **PASSED** (0 speaker overlap between train, validation, and test splits).

---

## 3. Baseline `LA_model.pth` Test Set Evaluation Results

- **Test Set Size**: `11` samples (`6` Real, `5` Synthetic)
- **Accuracy**: `45.45%`
- **Precision (Real)**: `0.0`
- **Recall (Real)**: `0.0`
- **F1 Score**: `0.0`
- **False Acceptance Rate (FAR)**: `0.0%`
- **False Rejection Rate (FRR)**: `100.0%`
- **Equal Error Rate (EER)**: `81.67%`

### Confusion Matrix

```
             Pred REAL   Pred SYNTHETIC
REAL             0            6
SYNTHETIC        0            5
```

---

## 4. Score Distribution Analysis

### Real Speech (`bona_fide`) Probabilities
- **Min**: `0.1477`
- **Max**: `0.2123`
- **Mean**: `0.1828`
- **Median**: `0.18`

### Synthetic Speech (`spoof`) Probabilities
- **Min**: `0.1556`
- **Max**: `0.2911`
- **Mean**: `0.242`
- **Median**: `0.2504`

---

## 5. Threshold Analysis (0.50 to 0.90)

| Threshold | Accuracy (%) | Precision | Recall | FAR (%) | FRR (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0.50` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.55` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.60` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.65` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.70` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.75` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.80` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.85` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |
| `0.90` | `45.45` | `0.0` | `0.0` | `0.0` | `100.0` |

*Notice: Threshold analysis is for diagnostic evaluation only. Production decision threshold remains unmodified at 0.50.*

---

## 6. Training Readiness Verdict

- **Dataset Sufficiency for Fine-Tuning**: **READY FOR PHASE 2C**
- **Speaker-Disjoint Enforced**: **YES**
- **No Training Executed**: **CONFIRMED** (Pipeline stopped before model mutation)
