# Combined Research Analysis

**Project:** Detecting Shortcut Learning in CNNs  
**Scope:** Retrospective synthesis of the prepared pilot dataset, baseline training, frozen-checkpoint counterfactual evaluation, and Grad-CAM outputs. No training, test evaluation, or dataset generation was rerun for this report.

## Executive summary

The pilot contains 50 class-specific sources per class and 164 composites. Training uses the preferred background for 31 of 34 samples per class (91.2%); validation and test each contain 48 composites arranged as 16 source groups with grass, sand, and snow variants.

The selected epoch-2 CNN achieves 50.0% accuracy on validation and 50.0% on the 48-image test set. In counterfactual testing, predictions changed in all 16 groups. All cattle were classified correctly on grass and snow but incorrectly on sand; all horses were classified correctly on sand but incorrectly on grass and snow. This pattern is strongly consistent with background-associated predictions in this pilot, but is not conclusive evidence of causal shortcut learning.

Grad-CAM activation was mostly outside the animal mask for all backgrounds: 93.02% for grass, 87.96% for sand, and 90.85% for snow. This is suggestive of broad background activation, but Grad-CAM is a coarse, model-dependent visualization and cannot establish why a prediction was made.

## A. Dataset construction

| Split | Cattle sources | Horse sources | Distinct underlying COCO image IDs | Composite images |
|---|---:|---:|---:|---:|
| Train | 34 | 34 | 65 | 68 |
| Validation | 8 | 8 | 15 | 48 |
| Test | 8 | 8 | 15 | 48 |
| **Total** | **50** | **50** | **95** | **164** |

The 100 class-specific source entries correspond to 95 distinct COCO image IDs because five images contain both cattle and horse annotations. Three such IDs occur in training, one in validation, and one in test. Metadata confirms that **no source image ID crosses dataset splits**. The different class annotations from a shared image stay in the same split.

### Background distributions

| Split / class | Grass | Sand | Snow |
|---|---:|---:|---:|
| Train / cattle | 31 | 2 | 1 |
| Train / horse | 2 | 31 | 1 |
| Validation / cattle | 8 | 8 | 8 |
| Validation / horse | 8 | 8 | 8 |
| Test / cattle | 8 | 8 | 8 |
| Test / horse | 8 | 8 | 8 |

The intended 90% preferred-background allocation is rounded to an integer per 34-source class: `round(34 × 0.90) = round(30.6) = 31`. Thus the realized rate is **31/34 = 91.18%**, with **3/34 = 8.82%** on alternatives. Cattle prefer grass and horses prefer sand. Validation and test are balanced at 8 examples per class-background cell.

Every validation and test animal source group contains exactly three composites, one for each of grass, sand, and snow. The variants share source image, source annotation, class label, group ID, and split.

## B. CNN training

The baseline is a small CNN trained from scratch. It has four 3×3 convolutional layers with channel widths 16, 32, 64, and 96; ReLU activations; 2×2 max pooling after the first three convolutional blocks; adaptive average pooling; and a two-class linear output. Images are RGB, 224×224, scaled to `[0,1]`.

Training configuration: seed 42, Adam with learning rate 0.001, cross-entropy loss, batch size 8, CPU with two threads, and early stopping with patience 6 (configured maximum: 30 epochs). No augmentation was used. Only training data was used for parameter updates; validation loss selected the checkpoint.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.6952 | 50.00% | 0.6935 | 50.00% |
| 2 | 0.6913 | 45.59% | **0.6935** | 50.00% |
| 3 | 0.6698 | 79.41% | 0.6943 | 47.92% |
| 4 | 0.5937 | 85.29% | 0.7069 | 45.83% |
| 5 | 0.4673 | 83.82% | 0.9459 | 50.00% |
| 6 | 0.3507 | 86.76% | 0.8098 | 45.83% |
| 7 | 0.2696 | 94.12% | 0.9782 | 50.00% |
| 8 | 0.2924 | 89.71% | 0.8847 | 52.08% |

Epoch 2 had the lowest validation loss and is the saved best checkpoint. Its validation metrics are: **50.00% accuracy**, 0.5000 macro precision, 0.5000 macro recall, and 0.4921 macro F1. The validation confusion matrix (actual rows; predicted columns; cattle then horse) is `[[15, 9], [15, 9]]`.

Training stopped after epoch 8 when validation loss had failed to improve for six consecutive epochs. The decreasing training loss and rising training accuracy, alongside validation loss worsening after epoch 2 and validation accuracy remaining near chance, are evidence consistent with overfitting. The tiny dataset makes this an indicative pattern rather than a precise estimate of generalization.

## C. Counterfactual test evaluation

The frozen epoch-2 checkpoint was evaluated on all 48 test composites only. Test accuracy is **24/48 = 50.00%**; macro precision is 0.5000, macro recall 0.5000, and macro F1 0.4857. The test confusion matrix is `[[16, 8], [16, 8]]` (actual rows; predicted columns; cattle then horse).

| Metric | Cattle | Horse |
|---|---:|---:|
| Samples | 24 | 24 |
| Accuracy / recall | 66.67% | 33.33% |
| Precision | 50.00% | 50.00% |
| F1 | 57.14% | 40.00% |

| Background | Correct | Samples | Accuracy |
|---|---:|---:|---:|
| Grass | 8 | 16 | 50.00% |
| Sand | 8 | 16 | 50.00% |
| Snow | 8 | 16 | 50.00% |

The aggregate background accuracies mask a systematic class-by-background pattern:

| True class | Grass | Sand | Snow |
|---|---:|---:|---:|
| Cattle | 8/8 (100%) | 0/8 (0%) | 8/8 (100%) |
| Horse | 0/8 (0%) | 8/8 (100%) | 0/8 (0%) |

For cattle, accuracy is 100% on its preferred grass background and 50% across the 16 alternative variants. For horses, accuracy is 100% on preferred sand and 0% on alternatives. Across all samples, the model predicted cattle for every grass and snow variant and horse for every sand variant.

All **16/16 source groups (100%)** changed prediction at least once across their three variants; zero groups had identical predictions. Of the 48 pairwise comparisons among the three variants within each group, 32 predictions differed: **32/48 = 66.67% pairwise flip rate**. No group was correct on all three backgrounds or wrong on all three; all 16 groups had mixed correctness.

The group-change rate and pairwise flip rate are different summaries: the former asks whether any variant changes within a group, while the latter counts differing pairs. Evaluation variants share the same underlying animal image and are correlated, so the 48 composites must not be interpreted as 48 independent source observations.

## D. Grad-CAM

Grad-CAM was generated for the same 48 test composites using the frozen epoch-2 model’s final convolutional layer, `SmallCNN.features[9]`. Segmentation-aligned animal regions were derived from the COCO masks transformed to composite coordinates; the transformed alpha mask was thresholded at 128. For each image, positive min-max-normalized predicted-class activation was apportioned inside and outside that region.

| Background | Mean activation inside animal | Mean activation outside animal |
|---|---:|---:|
| Grass | 6.98% | 93.02% |
| Sand | 12.04% | 87.96% |
| Snow | 9.15% | 90.85% |

Inside-animal activation was 5.07 percentage points higher for sand than grass and 2.89 points higher for sand than snow. Activation was predominantly outside the animal mask for all three backgrounds.

These are descriptive localization values, not causal measurements. Grad-CAM is spatially coarse and class-conditional; its activation can reflect context, animal features, compositing boundaries, or interactions among them. Normalized activation shares do not directly measure feature importance, and low inside-mask shares do not prove background reliance.

## Scientific interpretation and limitations

### Findings consistent with background shortcut learning

- The model’s predictions follow a simple background-associated rule in this test set: grass/snow map to cattle and sand maps to horse.
- Preferred-background accuracy is substantially higher than alternative-background accuracy for both classes (cattle: 100% vs 50%; horse: 100% vs 0%).
- Every source group changes prediction across counterfactual backgrounds, while Grad-CAM activation is predominantly outside the animal masks.

Together, these results are strongly **consistent with background sensitivity and possible shortcut use in this pilot**. They do not prove that the CNN causally relies on the background; the observed pattern could also reflect the small sample, animal/background compositing artifacts, dataset-specific image content, or other confounds.

### What cannot be concluded

- This pilot does not establish performance on a representative population of cattle or horses or generalization to natural images.
- It cannot isolate background causality from every correlated factor, nor show that background is the only signal the model uses.
- A single small train/validation/test split and one training run do not quantify uncertainty or seed-to-seed variability.
- The test has only eight animal source groups per class. Three variants of one animal are repeated, correlated measurements—not independent animals. Treating all 48 composites as independent would overstate the effective sample size.
- Grad-CAM is an explanatory visualization, not a faithful causal attribution test; the activation shares depend on layer, predicted class, mask transformation, and normalization.

The findings should therefore be reported as observations from a controlled pilot, not as conclusive proof of shortcut learning.

## Consistency audit

The dataset summary agrees with the 164 metadata rows for split sizes, class counts, and background counts. Metadata has no source image ID assigned to more than one split; every validation and test group has the expected three backgrounds. Test metrics, the test confusion matrix, per-background CSV, counterfactual summary, and raw predictions agree. Grad-CAM predictions match the test evaluator’s predictions, confidence, and correctness for all 48 images when joined by counterfactual group ID and background.

The validation metrics agree with the epoch-2 history row and validation confusion matrix. The validation JSON is for the restored **best epoch-2 checkpoint**, whereas the final history row is epoch 8; comparing those values as though they describe the same model would be misleading, but this is not an inconsistency.

No material numerical inconsistencies were found. The source-ID overlap across class annotations and the distinct denominators for group-change and pairwise flip rates are documented above.

## Source artifacts and generated report files

Source artifacts inspected:

- `data/pilot/dataset/dataset_summary.json`
- `data/pilot/dataset/metadata.csv`
- `experiments/baseline_cnn/training_history.csv`
- `experiments/baseline_cnn/validation_metrics.json`
- `experiments/baseline_cnn/confusion_matrix.csv`
- `experiments/counterfactual_evaluation/test_metrics.json`
- `experiments/counterfactual_evaluation/counterfactual_summary.json`
- `experiments/counterfactual_evaluation/background_accuracy.csv`
- `experiments/counterfactual_evaluation/predictions.csv`
- `experiments/gradcam_test/gradcam_summary.json`
- `experiments/gradcam_test/gradcam_predictions.csv`

Generated files:

- Full report: [`combined_research_report.md`](./combined_research_report.md)
- Machine-readable results: [`combined_results.json`](./combined_results.json)
- Concise summary: [`executive_summary.md`](./executive_summary.md)
