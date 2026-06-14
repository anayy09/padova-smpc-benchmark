"""Synthetic approximation of the UVA/Padova S2017 cohort.

This module creates 100 adult, 100 adolescent, and 100 pediatric virtual
subjects with parameter statistics roughly consistent with the published S2017
populations. The data are *synthetic* and do not use any proprietary
information, but they allow the rest of the pipeline to mimic the cohort size
and variability described in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class SubjectStats:
    """Distribution descriptors for a cohort."""

    bw_range: Tuple[float, float]
    si_range: Tuple[float, float]
    basal_range: Tuple[float, float]
    cr_range: Tuple[float, float]
    isf_range: Tuple[float, float]
    total_daily_insulin_range: Tuple[float, float]


# Approximate ranges for S2017 (sourced from public summaries)
COHORT_STATS: Dict[str, SubjectStats] = {
    "adult": SubjectStats(
        bw_range=(60.0, 100.0),
        si_range=(5.0e-5, 1.5e-4),
        basal_range=(0.7, 1.5),  # U/hr
        cr_range=(8.0, 18.0),    # g/U
        isf_range=(25.0, 60.0),  # mg/dL per U
        total_daily_insulin_range=(30.0, 70.0),
    ),
    "adolescent": SubjectStats(
        bw_range=(45.0, 80.0),
        si_range=(4.0e-5, 1.3e-4),
        basal_range=(0.6, 1.3),
        cr_range=(7.0, 16.0),
        isf_range=(30.0, 70.0),
        total_daily_insulin_range=(35.0, 80.0),
    ),
    "child": SubjectStats(
        bw_range=(20.0, 50.0),
        si_range=(6.0e-5, 1.8e-4),
        basal_range=(0.2, 0.9),
        cr_range=(10.0, 25.0),
        isf_range=(40.0, 90.0),
        total_daily_insulin_range=(15.0, 45.0),
    ),
}


def _sample_uniform(low: float, high: float, size: int) -> np.ndarray:
    return np.random.uniform(low, high, size)


def _build_subject_dict(cohort: str, subject_id: int, stats: SubjectStats) -> Dict:
    """Create one synthetic subject."""

    return {
        "patient_id": subject_id,
        "cohort": cohort,
        "BW": float(np.random.uniform(*stats.bw_range)),
        "SI": float(np.random.uniform(*stats.si_range)),
        "basal_rate": float(np.random.uniform(*stats.basal_range)),
        "CR": float(np.random.uniform(*stats.cr_range)),
        "ISF": float(np.random.uniform(*stats.isf_range)),
        "TDI": float(np.random.uniform(*stats.total_daily_insulin_range)),
    }


def generate_population(seed: int = 42) -> Dict[str, List[Dict]]:
    """Generate the synthetic S2017-equivalent population.

    Returns a dictionary with keys "adult", "adolescent", and "child" where
    each entry is a list of 100 subject dictionaries.
    """

    np.random.seed(seed)
    population: Dict[str, List[Dict]] = {}

    for cohort_name, stats in COHORT_STATS.items():
        subjects: List[Dict] = []
        for idx in range(100):
            subject = _build_subject_dict(cohort_name, idx, stats)
            subjects.append(subject)
        population[cohort_name] = subjects

    return population


def get_subjects(cohort: str, seed: int = 42) -> List[Dict]:
    """Helper that returns subjects for a single cohort."""

    population = generate_population(seed=seed)
    return population[cohort]


def get_all_subjects(seed: int = 42) -> List[Dict]:
    """Concatenate all cohorts (300 subjects)."""

    population = generate_population(seed=seed)
    all_subjects: List[Dict] = []
    for cohort in ["adult", "adolescent", "child"]:
        all_subjects.extend(population[cohort])
    return all_subjects
