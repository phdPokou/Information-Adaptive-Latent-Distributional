# Information-Adaptive Latent Distributional Robustness for Climate-Transition Portfolio Choice

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch CUDA">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white" alt="NVIDIA CUDA">
  <img src="https://img.shields.io/badge/Reproducibility-Enabled-success" alt="Reproducible">
  <img src="https://img.shields.io/badge/Research-Code-blue" alt="Research Code">
</p>

## Overview

This repository contains the computational implementation accompanying the paper

> **Information-Adaptive Latent Distributional Robustness for Climate-Transition Portfolio Choice**

The code reproduces the two main computational components of the study:

1. the **Controlled Simulation Study**, designed to evaluate the analytical mechanisms of information-adaptive latent distributional robustness under a controlled data-generating process;
2. the **Empirical Results**, based on a recursive out-of-sample portfolio experiment with climate-transition, competing-state, robustness, placebo, and statistical-inference analyses.

The implementation follows the information structure of the paper. The observable state \(S_t\) is available at portfolio formation, whereas future latent severity remains uncertain. State dependence enters through the nominal latent law and the radius of the latent Kullback-Leibler ambiguity set. Portfolio transition exposure is evaluated separately and does not determine the ambiguity set.

---

## Repository Structure

```text
.
├── README.md
├── section4_controlled_simulation_q1pp_v5.py
├── section6_q1pp_empirical_v7_fixed.py
├── requirements.txt
│
├── Results_Section4_V5/
│   ├── figures/
│   ├── figure_csv/
│   ├── tables/
│   ├── seed_level/
│   ├── diagnostics/
│   └── config/
│
└── Section6_Q1PP_Results_V7_FIXED/
    ├── csv/
    ├── figures/
    ├── tables/
    └── ...
```

Output directories are created automatically when the corresponding scripts are executed.

---

## Computational Environment

The implementation uses Python, PyTorch, NumPy, pandas, SciPy, and Matplotlib.

A CUDA-enabled NVIDIA GPU is recommended for the full experiments. Both scripts can nevertheless fall back to CPU execution where applicable.

### Main dependencies

```text
Python >= 3.10
numpy
pandas
scipy
matplotlib
torch
```

Create an isolated environment, for example:

```bash
python -m venv .venv
```

On Windows:

```cmd
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

PyTorch should be installed using the build appropriate for the local CUDA installation. Verify the installation with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# Reproducing the Computational Results

## 1. Controlled Simulation Study

The controlled simulation corresponds to the paper section **Controlled Simulation Study**.

The implementation preserves the principal structural restrictions of the theoretical model:

\[
S_t
\longmapsto
\left(
q_{\Phi_t(S_t)},
\Gamma_t(S_t)
\right),
\]

with the ambiguity radius remaining decision independent. In particular, the code does not implement an ambiguity radius of the form \(\Gamma_t(S_t,C_t(x))\).

The simulation uses multiple random seeds, common random numbers for paired comparisons, numerical optimization diagnostics, and CSV-backed figures and tables.

### Full CUDA run

From the repository root, execute:

```bash
python section4_controlled_simulation_q1pp_v5.py --output Results_Section4_V5 --use-cuda
```

The program automatically detects CUDA availability and records the computational environment used for the run.

### CPU execution

To force CPU execution:

```bash
python section4_controlled_simulation_q1pp_v5.py --output Results_Section4_V5 --cpu
```

### Smoke test

Before launching the complete experiment, the implementation can be checked using the reduced smoke-test configuration:

```bash
python section4_controlled_simulation_q1pp_v5.py --output Results_Section4_V5_Smoke --use-cuda --smoke-test
```

The smoke test reduces the number of seeds, states, Monte Carlo draws, and optimization iterations. It is intended to verify the computational pipeline rather than reproduce the final numerical results.

### Simulation outputs

The full run creates:

```text
Results_Section4_V5/
├── figures/
├── figure_csv/
├── tables/
├── seed_level/
├── diagnostics/
└── config/
```

The `figure_csv/` directory contains the numerical source data underlying the generated figures.

The `seed_level/` directory retains seed-level results used for repeated-simulation summaries.

The `diagnostics/` directory contains numerical and optimization checks, including the execution log.

The `config/` directory records the simulation configuration and computational environment. In particular, the code exports the Python, PyTorch, NumPy and pandas versions, the selected device, CUDA availability, and GPU information when available.

---

## 2. Empirical Results

The empirical pipeline corresponds to the paper section **Empirical Results**.

The implementation uses strict recursive timing: information dated after a portfolio formation date is excluded from that decision. The climate-transition state conditions the nominal latent law, while the future latent severity remains unresolved at portfolio formation.

The empirical pipeline additionally implements internal audit checks. Required empirical inputs are not silently replaced when unavailable.

### Required input directories

Two input locations are distinguished:

```text
Univers/
```

contains the empirical asset-return and factor data used by the portfolio experiment, while

```text
Z_t/
```

contains the validated state or climate-policy-uncertainty artifacts required to construct the climate-transition information state.

The empirical pipeline searches for the expected universe files, factor data, and, where available, dated transition-exposure scores. Both empirical universes are expected to contain 20 assets.

Users reproducing the analysis on another machine should replace the example paths below with their local paths.

### Full empirical run with CUDA

On Windows Command Prompt:

```cmd
python section6_q1pp_empirical_v7_fixed.py ^
    --data-dir "C:/Users/fredy/Downloads/Univers" ^
    --state-dir "C:/Users/fredy/Downloads/Z_t" ^
    --output-dir "C:/Users/fredy/Downloads/Section6_Q1PP_Results_V7_FIXED" ^
    --use-cuda ^
    --resume-v5 ^
    --resume-v5-dir "C:/Users/fredy/Downloads/Section6_Q1PP_Results_V6_FULL"
```

The option

```text
--resume-v5
```

instructs the V7 pipeline to reuse completed V5 state-placebo outputs instead of recomputing the corresponding A3 experiments.

The associated directory is supplied through:

```text
--resume-v5-dir
```

This option should therefore only be used when the specified directory contains the completed V5 outputs required by the V7 pipeline.

### Full run without V5 reuse

To recompute the pipeline without loading previous V5 state-placebo outputs:

```cmd
python section6_q1pp_empirical_v7_fixed.py ^
    --data-dir "C:/Users/fredy/Downloads/Univers" ^
    --state-dir "C:/Users/fredy/Downloads/Z_t" ^
    --output-dir "C:/Users/fredy/Downloads/Section6_Q1PP_Results_V7_FIXED" ^
    --use-cuda
```

### Fast diagnostic run

A reduced run can be used to verify the empirical pipeline:

```cmd
python section6_q1pp_empirical_v7_fixed.py ^
    --data-dir "C:/Users/fredy/Downloads/Univers" ^
    --state-dir "C:/Users/fredy/Downloads/Z_t" ^
    --output-dir "C:/Users/fredy/Downloads/Section6_Q1PP_FAST" ^
    --fast ^
    --use-cuda
```

The `--fast` option reduces several computational settings and is intended for implementation checks rather than reproduction of the final reported results.

---

## Empirical Data Requirements

The empirical script searches the data directory for the return universes and common factor data. The expected canonical filenames include:

```text
Universe_A_log_returns.csv
Universe_B_log_returns.csv
Z_factors_daily_common.csv
```

Alternative filename variants recognized by the script are also supported.

The climate-transition state requires the relevant EUA and narrow climate-policy-uncertainty information. Validated state artifacts can be supplied separately through:

```text
--state-dir
```

The pipeline deliberately does not construct an unvalidated replacement when the required climate-policy-uncertainty input is unavailable.

If dated asset-level transition-exposure scores are supplied, recognized filenames include:

```text
transition_exposure_scores.csv
asset_transition_exposure.csv
c_t_scores.csv
```

These exposure scores are used only for the corresponding exposure analyses. Economic asset classifications are not substituted for missing \(c_{i,t}\) scores.

---

## Reproducibility and Numerical Auditing

Reproducibility is built into both computational pipelines.

### Controlled simulation

The simulation uses a predefined collection of random seeds and records:

```text
configuration
Python version
platform
PyTorch version
NumPy version
pandas version
CUDA availability
selected computational device
GPU model, when available
```

The simulation also exports numerical diagnostics and the CSV source data underlying figures and tables.

### Empirical analysis

The empirical pipeline uses a fixed master seed and writes a provenance file containing the configuration and computational environment.

The implementation checks, among other conditions:

* availability of required input files;
* admissibility of the climate-transition state;
* portfolio feasibility;
* temporal ordering between portfolio formation and realized returns;
* availability of required state inputs;
* CUDA availability when GPU execution is requested.

These checks are intended to detect implementation or data-integrity failures. A numerical `PASS` status should not be interpreted as a statistical hypothesis test or as evidence supporting an economic hypothesis.

---

## CUDA

GPU execution is enabled with:

```text
--use-cuda
```

When CUDA is requested and available, PyTorch operations used by the latent robust optimization routines are executed on the GPU.

CUDA availability can be checked before running the experiments:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

and the detected GPU can be inspected with:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

The exact PyTorch installation command depends on the CUDA version available on the reproduction machine.

---

## Reproduction Workflow

For a clean reproduction, the recommended sequence is:

```text
1. Clone the repository.
2. Create and activate an isolated Python environment.
3. Install the required dependencies.
4. Verify PyTorch and CUDA availability.
5. Run the Controlled Simulation Study smoke test.
6. Run the full Controlled Simulation Study.
7. Place the empirical input files in the required data directories.
8. Run the empirical pipeline in fast mode as an integrity check.
9. Run the full empirical specification.
10. Inspect the generated logs, diagnostics, CSV files, tables, and figures.
```

For the final results reported in the paper, reduced `--smoke-test` and `--fast` configurations should not be substituted for the full runs.

---

## Methodological Scope

The repository implements the information-adaptive latent robustness architecture studied in the paper. In particular,

\[
S_t
\mapsto
q_{\Phi_t(S_t)}
\]

controls the state-indexed nominal latent law, while

\[
S_t
\mapsto
\Gamma_t(S_t)
\]

controls the state-indexed KL ambiguity radius.

Portfolio transition exposure,

\[
C_t(x)=c_t^\top x,
\]

is kept separate from the ambiguity specification. Consequently, conditional on a fixed information state, feasible portfolios face the same latent ambiguity set. The implementation should therefore not be interpreted as a model of decision-dependent ambiguity.

---

## Citation

If you use this repository in academic work, please cite the accompanying paper:

```bibtex
@article{PokouSadefoInformationAdaptive,
  title   = {Information-Adaptive Latent Distributional Robustness for Climate-Transition Portfolio Choice},
  author  = {Pokou, Fr{\'e}dy and Sadefo Kamdem, Jules},
  note    = {Manuscript}
}
```

The bibliographic record should be updated with the journal, year, volume, pages, and DOI once these become available.

---

## Authors

**Frédy Pokou**  
**Jules Sadefo Kamdem**

---

## License

Please consult the repository `LICENSE` file for the terms governing reuse of the source code.

---

## Reproducibility Note

The scripts save computational metadata and numerical diagnostics to facilitate independent reproduction. Exact numerical agreement can nevertheless depend on the software stack, hardware, CUDA and PyTorch versions, and numerical behavior of the optimization routines. Reproduction claims should therefore be evaluated together with the saved configuration, provenance files, seed-level outputs, and diagnostics generated by each run.
