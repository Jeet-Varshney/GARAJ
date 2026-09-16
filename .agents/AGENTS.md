# GARAJ Workspace Guidelines & Behavioral Rules

## Accuracy & Model Integrity Mandate
- **Priority**: Preserve and prioritize detection accuracy as top priority across all ML, data, and software tasks.
- **No Synthetic UI Hacks**: Never sacrifice model accuracy or alter genuine model outputs to fit UI aesthetics.
- **Zero Fake / Heuristic Predictions**: Never use random, fake, hardcoded, or heuristic predictions.
- **Strict Class Mapping**: Never invert REAL/SYNTHETIC predictions without verifying the model's actual class mapping against authoritative test scripts.
- **Validation & Metrics**: Always evaluate changes against a proper validation/test set. Report accuracy, precision, recall, F1, FAR, FRR, and EER whenever applicable.
- **Data Leakage Prevention**: Enforce strict speaker-disjoint train/validation/test splits.
- **Multilingual Evaluation**: Maintain separate evaluation for Hindi, Hinglish, and English wherever data exists.
- **Domain Performance**: Test consumer-microphone/domain performance separately from benchmark performance.
- **Empirical Proof**: Do not claim improved accuracy unless empirical evaluation demonstrates it.
- **Production Checkpoint Retention**: Keep the best validated model/checkpoint (`LA_model.pth`) rather than automatically replacing the production model.
