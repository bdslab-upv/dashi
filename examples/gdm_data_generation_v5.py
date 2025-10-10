# -*- coding: utf-8 -*-
"""
Synthetic GDM Dataset Generation with Advanced Concept Shift.

This script generates a synthetic dataset simulating the diagnosis of Gestational
Diabetes Mellitus (GDM). It features an advanced concept shift that models the
gradual correction of a systemic, asymmetric data entry error.

This scenario includes several layers of realism:
- **Asymmetric Errors:** The underlying data error is more likely to affect one
  class over the other (e.g., GDM cases are mislabeled more often than non-GDM).
- **Non-Linear Correction:** The error correction rate follows a sigmoid (S-shaped)
  curve, simulating a realistic adoption pattern for a fix (slow start, rapid
  correction, and a final tail-off).
- **Stochasticity:** The number of errors per month is not deterministic but is
  drawn from a random distribution, reflecting natural monthly variations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# ==============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION WITH ADVANCED CONCEPT SHIFT
# ==============================================================================
# This section creates the core dataset with a more realistic concept shift.
# The underlying data structure is a Gaussian Mixture Model (GMM), where each
# class is represented by a distinct Gaussian distribution in a
# multi-dimensional feature space.

# --- 1.1: Simulation Parameters ---

# Set a seed for the random number generator to ensure reproducibility.
# Anyone running this script with the same seed will get the exact same dataset.
np.random.seed(42)

# Define the simulation time frame: a monthly frequency from Jan 2021 to Dec 2024.
dates: pd.DatetimeIndex = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
months: int = len(dates)

# Define the dimensionality of our feature space. We are simulating 5 key predictors.
d: int = 5
# Define the covariance matrix for the Gaussian distributions. `np.eye(d)` creates
# an identity matrix, signifying that the base features are uncorrelated.
# This is a simplifying assumption; in reality, features might have some correlation.
Sigma: np.ndarray = np.eye(d)

# Mean vectors for the two Gaussian components, representing two latent patient
# "endotypes" or profiles.
# Component A represents a profile with lower glycemic values, typical of a
# non-GDM patient (euglycemic).
mu_A: np.ndarray = np.array([-1.25, -0.5, 0.1, 0.0, 0.0])
# mu_A: np.ndarray = np.array([-1.5, -0.75, 0.1, 0.0, 0.0])
# Component B represents a profile with higher glycemic values, typical of a
# GDM patient (hyperglycemic).
mu_B: np.ndarray = np.array([1.25, 0.5, -0.1, 0.0, 0.0])
# mu_B: np.ndarray = np.array([2, 0.75, -0.1, 0.0, 0.0])


def get_flipping_probabilities(
    t_idx: int,
    max_rate_y0: float = 0.2, # 0.25, 0.2
    max_rate_y1: float = 0.94 # 0.65 0.94
) -> Tuple[float, float]:
    """Calculates the time-dependent, asymmetric label flipping probabilities.

    This function is the core of the concept shift mechanism. It simulates a
    scenario where a systemic error (e.g., a faulty lab machine or a data
    entry bug) is gradually fixed over time. The probability of an error
    (a "flip") is high at the beginning and decays to nearly zero by the end.

    The decay follows a sigmoid function, which models a realistic adoption
    curve for a fix: a slow start, a period of rapid implementation, and then a
    final phase of ironing out the last issues.

    The error is also asymmetric: the maximum error rate for the GDM class (`y=1`)
    is significantly higher than for the non-GDM class (`y=0`), modeling a bias
    in the error mechanism.

    Args:
        t_idx (int): The zero-based index for the current time step (month).
        max_rate_y0 (float): The peak error rate for the y=0 class (non-GDM).
        max_rate_y1 (float): The peak error rate for the y=1 class (GDM).

    Returns:
        Tuple[float, float]: A tuple containing the flipping probability for
                             class 0 and class 1 for the given time step.
    """
    # Handle the edge case of a single time step to avoid division by zero.
    if months == 1:
        return max_rate_y0, max_rate_y1

    # Normalize the time index to the range [-7, 7]. This mapping is crucial.
    # A standard normalization to [0, 1] would result in a very gradual sigmoid
    # curve. By stretching it to a wider range like [-7, 7], we effectively
    # use the steepest part of the sigmoid function, creating a more rapid
    # and pronounced transition period for the error correction.
    scaled_time = 14.0 * (t_idx / (months - 1)) - 7.0

    # The sigmoid function maps the scaled time to a value between 0 and 1.
    # As `t_idx` goes from 0 to `months-1`, `scaled_time` goes from -7 to 7,
    # and `sigmoid_decay` goes from near 0 to near 1.
    sigmoid_decay = 1.0 / (1.0 + np.exp(-scaled_time))

    # The flipping probability is inversely related to the sigmoid decay.
    # At t=0, `sigmoid_decay` is ~0, so `p_flip` is at its maximum.
    # At t=`months-1`, `sigmoid_decay` is ~1, so `p_flip` is ~0.
    p_flip_y0 = max_rate_y0 * (1.0 - sigmoid_decay)
    p_flip_y1 = max_rate_y1 * (1.0 - sigmoid_decay)

    return p_flip_y0, p_flip_y1


# --- 1.2: Monthly Data Generation Loop ---
rows: List[pd.DataFrame] = []
# We iterate through each month in our defined date range to generate a batch of data.
for t, dt in enumerate(dates):
    # Simulate realistic variation in patient volume per month.
    rows_per_month: int = np.random.randint(1800, 2201)
    batch: str = dt.strftime("%Y-%m-01")

    # --- Generate outcome `y` with a seasonal prior shift ---
    # This simulates how GDM prevalence can vary with seasons (e.g., due to
    # lifestyle changes). The `np.cos` function creates a smooth, cyclical shift.
    n: int = rows_per_month
    seasonal_factor = np.cos((dt.month - 7) * (2 * np.pi / 12)) # Peaks in summer
    base_prevalence = 0.175
    amplitude = 0.075
    target_prevalence = base_prevalence + amplitude * seasonal_factor
    noise = np.random.uniform(-0.025, 0.025) # Add small random monthly noise
    prevalence: float = target_prevalence + noise

    # Create the ground-truth labels based on the calculated prevalence.
    n1: int = int(n * prevalence)
    n0: int = n - n1
    y: np.ndarray = np.array([0] * n0 + [1] * n1)
    np.random.shuffle(y) # Randomize the order of labels.

    # --- Advanced Concept Shift via Asymmetric, Stochastic Flipping ---
    # Get the error probabilities for the current month `t`.
    p_flip_y0_t, p_flip_y1_t = get_flipping_probabilities(t)

    # Initially, we assume a perfect world: all y=0 patients belong to the
    # euglycemic component (A) and all y=1 patients to the hyperglycemic (B).
    comps: np.ndarray = np.zeros(n, dtype=bool)  # `False` corresponds to component B
    idx0: np.ndarray = np.where(y == 0)[0]
    idx1: np.ndarray = np.where(y == 1)[0]
    comps[idx0] = True  # `True` corresponds to component A

    # Now, we introduce the error. Instead of flipping a fixed percentage, we
    # simulate a more realistic stochastic process. For each class, the number of
    # records to flip is drawn from a binomial distribution. This models the
    # randomness of which specific records are affected by the error each month.
    n0_to_flip = np.random.binomial(n=len(idx0), p=p_flip_y0_t)
    n1_to_flip = np.random.binomial(n=len(idx1), p=p_flip_y1_t)

    # Randomly select and flip records for the y=0 class (A -> B).
    # A flip means a patient with a non-GDM label (`y=0`) is now associated
    # with the hyperglycemic feature profile (`component B`), creating a mismatch.
    if n0_to_flip > 0:
        flip_indices_0: np.ndarray = np.random.choice(idx0, size=n0_to_flip, replace=False)
        comps[flip_indices_0] = False # Flip from A (True) to B (False)

    # Randomly select and flip records for the y=1 class (B -> A).
    # A flip means a patient with a GDM label (`y=1`) is now associated
    # with the euglycemic feature profile (`component A`).
    if n1_to_flip > 0:
        flip_indices_1: np.ndarray = np.random.choice(idx1, size=n1_to_flip, replace=False)
        comps[flip_indices_1] = True # Flip from B (False) to A (True)

    # Generate feature data `X` from the GMM based on the (now potentially flipped)
    # component assignments. `np.where` efficiently selects the correct mean vector
    # (`mu_A` or `mu_B`) for each patient.
    mus: np.ndarray = np.where(comps[:, None], mu_A[None, :], mu_B[None, :])
    X: np.ndarray = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n) + mus

    # --- Introduce a simple Covariate Shift ---
    # This simulates a gradual, linear change in population characteristics over time,
    # independent of the concept shift. Here, we model a slow increase in average
    # BMI and Maternal Age across the entire patient population.
    time_trend: float = -1.0 + 2.0 * (t / (months - 1)) # Linearly from -1 to 1
    time_shift_magnitude = 0.25
    time_shift = time_trend * time_shift_magnitude
    X[:, 3] += time_shift  # Apply shift to the 4th feature (BMI)
    X[:, 4] += time_shift  # Apply shift to the 5th feature (Maternal Age)

    # --- Assemble the monthly DataFrame ---
    df_m: pd.DataFrame = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(d)])
    df_m["y"] = y
    df_m["batch_date"] = batch
    df_m["batch_index"] = t + 1
    # We store the ground-truth flipping probabilities. This is invaluable for
    # evaluating drift detection algorithms later on.
    df_m["p_flip_y0"] = p_flip_y0_t
    df_m["p_flip_y1"] = p_flip_y1_t
    df_m["component"] = np.where(comps, "A", "B")
    rows.append(df_m)

# --- 1.3: Final Assembly and Feature Creation ---
# Concatenate all the monthly DataFrames into a single, comprehensive dataset.
ds: pd.DataFrame = pd.concat(rows, ignore_index=True)

# Create helper columns for analysis and modeling, such as splitting the data
# into two distinct "eras" for training and testing model robustness.
ds["split"] = np.where(ds["batch_index"] <= months // 2, "train_era", "test_era")
ds["rolling3"] = ((ds["batch_index"] - 1) // 3) + 1 # Groups months into quarters


# ==============================================================================
# SECTION 2: MAP Z-SCORES TO CLINICAL UNITS
# ==============================================================================
# This section de-normalizes the generated data from standard normal units
# (z-scores) into clinically meaningful values. This makes the dataset
# far more realistic and interpretable.

# Define the real-world mean and standard deviation for each feature.
MEANS: dict[str, float] = {
    "fasting_glucose_mgdl": 85.0, "oneh_glucose_mgdl": 160.0,
    "hba1c_percent": 5.3, "bmi_kg_m2": 27.0, "maternal_age_years": 31.0,
}
SDS: dict[str, float] = {
    "fasting_glucose_mgdl": 10.0, "oneh_glucose_mgdl": 30.0,
    "hba1c_percent": 0.4, "bmi_kg_m2": 6.0, "maternal_age_years": 6.0,
}

# Create new columns for the original z-scores for reference.
ds["z_fasting_glucose"] = ds["x1"]
ds["z_1h_glucose"] = ds["x2"]
ds["z_hba1c"] = ds["x3"]
ds["z_bmi"] = ds["x4"]
ds["z_maternal_age"] = ds["x5"]

# Apply the inverse transformation: clinical_value = z_score * SD + Mean
ds["fasting_glucose_mgdl"] = ds["z_fasting_glucose"] * SDS["fasting_glucose_mgdl"] + MEANS["fasting_glucose_mgdl"]
ds["oneh_glucose_mgdl"] = ds["z_1h_glucose"] * SDS["oneh_glucose_mgdl"] + MEANS["oneh_glucose_mgdl"]
ds["hba1c_percent"] = ds["z_hba1c"] * SDS["hba1c_percent"] + MEANS["hba1c_percent"]
ds["bmi_kg_m2"] = ds["z_bmi"] * SDS["bmi_kg_m2"] + MEANS["bmi_kg_m2"]
ds["maternal_age_years"] = ds["z_maternal_age"] * SDS["maternal_age_years"] + MEANS["maternal_age_years"]

# Round the values to a realistic number of decimal places.
ds["fasting_glucose_mgdl"] = ds["fasting_glucose_mgdl"].round(0)
ds["oneh_glucose_mgdl"] = ds["oneh_glucose_mgdl"].round(0)
ds["hba1c_percent"] = ds["hba1c_percent"].round(1)
ds["bmi_kg_m2"] = ds["bmi_kg_m2"].round(1)
ds["maternal_age_years"] = ds["maternal_age_years"].round(0)


# ==============================================================================
# SECTION 3: FINAL DATASET CLEANUP AND EXPORT
# ==============================================================================
# This section prepares the final dataset for consumption by renaming columns
# to be more descriptive and selecting the final set of columns to be exported.

# Use more intuitive names for the final columns.
ds = ds.rename(columns={"y": "gdm_diagnosis", "component": "latent_component_AB"})
comp_map: dict[str, str] = {"A": "euglycemic_endotype", "B": "hyperglycemic_endotype"}
ds["glycemic_endotype"] = ds["latent_component_AB"].map(comp_map)

# --- 3.1: Define Final Output Columns ---
# We explicitly define the columns and their order for the final output file.
# This ensures a consistent and clean final product. We include metadata,
# the target, predictors, and the latent ground truth variables.
cols_out: List[str] = [
    # Metadata
    "batch_date",
    # Target Variable
    "gdm_diagnosis",
    # Clinical Predictors
    "fasting_glucose_mgdl", "oneh_glucose_mgdl", "hba1c_percent",
    "bmi_kg_m2", "maternal_age_years",
]

# --- 3.2: Export to CSV ---
# The final dataset is ready. This line is commented out but shows how you
# would save the selected columns to a CSV file.
out_path: str = "/mnt/data/gdm_advanced_shift_dataset_CLINICAL_UNITS.csv"
# ds[cols_out].to_csv(out_path, index=False)
ds = ds[cols_out]

# --- Final Output for Verification ---
print("Advanced dataset generation script updated successfully.")
print("Final DataFrame columns:", ds[cols_out].columns.tolist())
print("\nFirst 5 rows of the generated data:")
print(ds[cols_out].head())