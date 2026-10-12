# CEDA-T reproduction materials

This repository contains the available evaluation records and reference code for the CEDA-T study. The main reproducible workflow recalculates the statistics in manuscript Tables III–IX from saved case-level outputs. It also summarizes the five-case parameter and seed-stability results reported in the appendix.

## Run

Use Python 3.11 and NumPy 1.26.4 (the recorded validation environment). From the repository root:

```bash
python -B run_all.py --run-id check1
```

Use a new `--run-id` for each run. Results are written to `reproduced/<run-id>/`, `tables_III_IX/reproduced/`, and `c10_stability/reproduced/`. The script runs locally without training models, running simulations, or calling external services.

## Files

- `tables_III_IX/`: archived per-case measurements, table aggregation, confidence-interval calculations, and saved table checks.
- `c10_stability/`: saved five-case parameter/seed evaluations and their summary calculations.
- `source_reference/`: the available neural and BSM ranking source components; these are reference components, not a complete historical experiment runner.
- `data_generation/`: descriptions of the synthetic case-generation procedures and evaluation settings.
- `configs/`, `seeds/`, and `splits/`: recorded settings, seeds, and available data partitions.
- `REPRODUCIBILITY.md`: details of what is and is not independently regenerated.

## Scope

The table scripts reconstruct statistics from archived per-case records. They do not reproduce the original model training, neural attribution, data generation, or BSM plant runs from scratch. The original sample-level split indices for the historical Tables III, V, and IX are not available; saved split records for the later strict-protocol experiments are provided separately. The synthetic expert actions in SEEK-Sim are rule-generated and are not human expert ratings.
