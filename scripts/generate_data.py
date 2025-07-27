import pandas as pd
import numpy as np
import os

np.random.seed(1982)
n_samples = 1000

# Burst pressure model
coeffs = {
    "balloon_dia_mm": 15,
    "material_thickness_mm": -100,
    "stretch_ratio": 50,
    "polymer_modulus_mpa": 0.05,
    "bond_temp_C": 2,
    "cooling_rate_Cps": -1
}

# Define helper functions to generate simulated scripts
def truncated_normal(mean, std, min_val, max_val, size):
    data = np.random.normal(mean, std, size)
    return np.clip(data, min_val, max_val)

def sim_bimodal_dist(mean1, std1, mean2, std2, ratio, size) -> np.ndarray:
    n1 = int(size * ratio)
    n2 = size - n1
    data = np.concatenate([
        np.random.normal(mean1, std1, n1),
        np.random.normal(mean2, std2, n2)
    ])
    np.random.shuffle(data)
    return data

# Parameters for simulated scripts

balloon_dia_mm = np.random.choice([3.0, 4.0, 5.0], size=n_samples, p=[0.3, 0.5, 0.2])
material_thick_mm = truncated_normal(0.06, 0.01, 0.04, 0.08, n_samples)
stretch_ratio_mm = truncated_normal(4.5, 0.4, 3.5, 5.5, n_samples)
polymer_modulus_mpa = np.random.normal(1500, 200, n_samples)
bond_temp_C = sim_bimodal_dist(
    mean1=150, std1=5,
    mean2=185, std2=5,
    ratio=0.6, size=n_samples
)

cooling_rate_Cps = np.clip(
    np.random.lognormal(mean=2, sigma=0.3, size=n_samples),
    5, 20
)

# Put the parameters together
df = pd.DataFrame({
    "balloon_dia_mm": balloon_dia_mm,
    "material_thickness_mm": material_thick_mm,
    "stretch_ratio": stretch_ratio_mm,
    "polymer_modulus_mpa": polymer_modulus_mpa,
    "bond_temp_C": bond_temp_C,
    "cooling_rate_Cps": cooling_rate_Cps
})

# Generate targets
noise = np.random.normal(0, 5, n_samples)

df["burst_pressure_psi"] = (
    df["balloon_dia_mm"] * coeffs["balloon_dia_mm"] +
    df["material_thickness_mm"] * coeffs["material_thickness_mm"] +
    df["stretch_ratio"] * coeffs["stretch_ratio"] +
    df["polymer_modulus_mpa"] * coeffs["polymer_modulus_mpa"] +
    df["bond_temp_C"] * coeffs["bond_temp_C"] +
    df["cooling_rate_Cps"] * coeffs["cooling_rate_Cps"] +
    noise
)

df = df.round(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "simulated_data.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  # Create directory if needed

df.to_csv(OUTPUT_PATH, index=False)
print(f"Simulated Scripts generated and written to {OUTPUT_PATH}")
