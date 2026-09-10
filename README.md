## Contextual Features & Pipeline Integration 

### 1. Integration Overview
The contextual feature extraction module integrates upstream with:
* **Member 2 (Clean Pipeline):** Ingests standardized, cleaned raw events/interaction datasets.
* **Member 5 (Vocabularies & Final Merge):** Maps entity IDs to global vocabulary indices and merges contextual features into the final training dataset.
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

## Model Training & Pipeline Validation

## 1. Quick Verification (Dry-Run)
To verify shape compatibility and check for memory leaks without training fully:
```bash
python train.py --dry-run