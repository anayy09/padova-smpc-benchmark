"""
Formal train/test partition for hyperparameter selection.

Training set: first 30 adult subjects (indices 0-29, seed 42).
Test set: remaining 70 adult subjects + all 200 non-adult subjects.
The partition is defined once here and imported wherever needed so
that the split is never accidentally changed between scripts.
"""

TUNING_COHORT = 'adult'
TUNING_SCENARIO = 'standard'
TRAIN_SIZE = 30

TRAIN_SUBJECT_IDS = list(range(TRAIN_SIZE))        # adult indices 0-29
TEST_SUBJECT_IDS  = list(range(TRAIN_SIZE, 100))   # adult indices 30-99
