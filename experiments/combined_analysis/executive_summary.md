# Executive Summary

The controlled pilot contains **50 source annotations per class** (cattle and horse), represented by **95 distinct COCO image IDs** and 164 composites. No source image ID crosses splits. Training uses preferred backgrounds in 31/34 examples per class (**91.2%**, from rounding 90% × 34 to 31). Validation and test each have 8 sources per class, with grass, sand, and snow variants for every evaluation source group.

The small four-convolution CNN selected checkpoint epoch 2 by validation loss. Its validation accuracy was **50.0%**. Training ended at epoch 8 with 89.7% training accuracy but only 52.1% validation accuracy and higher validation loss, a pattern consistent with overfitting.

On the frozen checkpoint’s **48 test composites**, accuracy was **50.0%**. Cattle accuracy was 100% on grass and snow but 0% on sand; horse accuracy was 100% on sand but 0% on grass and snow. Predictions changed across all 16 counterfactual groups; 32 of 48 within-group variant pairs differed (**66.7% pairwise flip rate**).

Grad-CAM activation was mostly outside the animal mask: **93.02% grass, 87.96% sand, and 90.85% snow**. This and the counterfactual results are consistent with background sensitivity and possible shortcut use, but do **not** prove causal background reliance. Only eight source groups per class were tested, and variants from the same source are correlated.

Full analysis: [`combined_research_report.md`](./combined_research_report.md)  
Machine-readable results: [`combined_results.json`](./combined_results.json)
