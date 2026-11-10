# PVGIS irradiance series

The study evaluates four Saudi locations: Riyadh, Medina, Mecca and Dammam, using PVGIS records spanning 1 January 2005 to 31 December 2020. The predictor series are global irradiance (GI, W/m²), sun elevation (SE, degrees), ambient temperature (AT, °C), relative humidity (H, %) and wind velocity (WV, m/s).

A measured-data experiment uses a chronological CSV with columns `time,GI,SE,AT,H,WV`. The benchmark should be evaluated independently for each city, with all preprocessing fitted exclusively on the development period. The implementation keeps the final 20% of rows as a chronological test interval; validation uses the last 15% of the remaining 80%, a documented engineering choice.

The original PVGIS data files are not contained in this repository. `train.py --smoke` creates a small synthetic signal solely for software validation. Documentation of the chosen PVGIS endpoint, irradiance variable mapping, request settings and downloaded file checksums is required for a numerical replication study.

Official source: https://re.jrc.ec.europa.eu/pvg_tools/en/
