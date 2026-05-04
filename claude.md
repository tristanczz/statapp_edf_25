# statapp_edf_25 — EDF bias correction project

## Goal
Multivariate climate bias correction using CMIP6 data (historical + ssp585),
compared to observations (obs.csv). Variables: tas, sfcWind, rlds, rsds, huss, uas, vas.

## Methods to implement
1. CDF-t univariate (done in functions_g.py)
2. Optimal transport multivariate (skeleton in functions.py via POT library)
3. Schrödinger bridge (Sinkhorn regularised OT, extract dual potentials)

## Key development priorities
- Gaussian multivariate test: compare CDF-t vs OT vs SB on known distributions
- Correlation matrix recovery as dimension grows (d=1,2,4,6)
- Soft assignment barycenter for Schrödinger bridge instead of hard argmax
- Wasserstein distance and Frobenius error as comparison metrics

## Files
- functions_g.py: Gilles's functions (CDF-t, plotting)
- functions.py: shared functions (OT skeleton with transport_optimal_uniforme)
- MAIN_g.ipynb: Gilles's notebook
- data/: CMIP6_historical.csv, CMIP6_ssp585.csv, obs.csv (sep=';')