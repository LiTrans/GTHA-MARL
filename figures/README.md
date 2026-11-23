# Manuscript figures

## Plot an evaluation history

Plot the `monthly.csv` written by the evaluation command to see employment, participation and unemployment for that simulated history. Each rate is shown as a percentage. The title identifies the policy seed, history index and evaluation seed.

```bash
python figures/render.py --history demo/history42/monthly.csv --output demo/history-figure
```

This writes `labour_market_rates.svg` and `labour_market_rates.pdf`. The plot sorts the observations by calendar month and leaves gaps where months or individual rates are missing.

## Render figures from summary tables

From the repository root, generate two sets of illustrative scores and render the recruitment and monthly labour-market figures.

```bash
python figures/example.py demo/figure-inputs
python figures/render.py --inputs demo/figure-inputs --output demo/figures --figure Fig3 S9
```

The example draws invented values for five policy seeds. Its figures demonstrate the plotting conventions and file formats. The paper uses estimates calculated from project data.

For a complete set of study summaries, use the same command with your input directory and omit `--figure`. Select individual figures by their labels, for example `--figure Fig4 ED5 ED6`. Each selected figure produces an editable SVG and a vector PDF. Figures 1, 4 and 5 also require Inkscape on the executable path. All other figures use Python libraries.

## Input map

Each path below is relative to `--inputs`. Brace groups list separate files in the same folder. Standard plotting domains, calendar windows, outcome definitions and seed counts correspond to the study design.

| Figure | Input files |
| --- | --- |
| Fig. 1. Model framework | `spatial/census_2006_cma_locator.json` and the supplied `framework.svg` diagram |
| Fig. 2. Historical rates, controls and four criteria | `learning/historical_learning_source.csv`, `diagnostics/registered_criteria.csv` |
| Fig. 3. Recruitment and learning | `learning/recruitment_learning_source.csv` |
| Fig. 4. Commute and vacancy access | `spatial/{registered_acceptance_inference,spatial_2016_summary,vacancy_access_2016_summary}.json` |
| Fig. 5. Pandemic response | `spatial/{pandemic,observed}.csv` |
| Fig. 6. Tariff effects | `tariff/scores.zip` |
| ED1. Criteria by policy | `diagnostics/registered_criteria.csv` |
| ED2. Age-specific rates and criteria at both scales | `core/{age_stocks,registered_criteria}.csv` |
| ED3. Scale sensitivity | `core/{historical,historical_seeds,historical_reference}.csv`, `evaluation/HISTORICAL_RESIDENT_SEED_CURVES.csv` |
| ED4. Recruitment paths | `controls/MECHANISM_ANNUAL_CURVES.csv` |
| ED5. Offer-attribute sensitivity | `spatial/registered_acceptance_inference.json` |
| ED6. Spatial outcomes by seed | `diagnostics/detailed_spatial.csv`, `spatial/spatial_2016_summary.json` |
| ED7. Pandemic transitions | `core/mechanism_policy.csv`, `evaluation/paired_histories.json` |
| ED8. Tariff scenarios | `tariff/scores.zip` |
| ED9. Training diagnostics | `training/training_diagnostics_source.csv` |
| S1. Four pandemic conditions | `core/{forward,changes,observed}.csv` |
| S2. Paired pandemic effects | `core/paired.csv` |
| S3. Historical fit against controls | `controls/{MACRO_PAIRED_POLICY_CONTRASTS,MACRO_POLICY_SCORES}.csv` |
| S4, S6. Firm and worker-policy interventions | `firm/{macro_annual_policy,mechanism_policy}.csv`, `core/historical_reference.csv` |
| S5. Worker flows and wages | `controls/FLOW_WAGE_ANNUAL_CURVES.csv`, `wages/wages_summary.csv` |
| S7. Individual policy trajectories | `core/{historical_seeds,historical_reference,forward_seeds,observed}.csv` |
| S8. Unemployment durations | `core/{mechanism,age_duration,duration_bins}.csv` |
| S9. Unemployment exits and job capacity | `diagnostics/historical_capacity_and_exits.csv` |

The learning tables use one row per panel, configuration, seed, year and outcome. The observation series uses `LFS` and an empty seed field. The simulation configurations are `trained`, `untrained_random` and `untrained_heuristic`. For period summaries the year field is empty. Means weight the five policy seeds equally. Paired effects subtract the comparator within each policy seed before aggregation. For ED7, each matched history has a case, policy seed, replication index, evaluation seed and checkpoint hash. The transition table identifies the simulation observations with `dataset=model`.

Figures 2 and 3 can also build their learning tables from a full set of control scores. With the panel-specific source CSV omitted, the renderer reads the reference, annual policy curves, window estimates, paired contrasts and corresponding summaries from `controls/`. It checks the seed-level means and intervals against the exported summaries before drawing the figure.

The criteria file uses `case,policy_seed,metric,full_11_mean_pct,full_11_ci_half_pp,last_3_mean_pct,last_3_ci_half_pp,final_classification`. The two intervals describe variation across histories within a policy. The training table contains `policy_seed,stage,stage_update,cumulative_update,worker/mean_reward_raw,firm/mean_reward_raw,worker_kl,firm_kl`.

The tariff ZIP contains the scoring tables `PANEL_EFFECTS.csv`, `MONTHLY_MACRO_SERIES.csv`, `ROLLUP_SUPPORT.csv`, `ROLLUP_POLICY_EFFECTS.csv`, `ROLLUP_PANEL_EFFECTS.csv` and `PRODUCT_GROUP_REPRESENTATION.csv` at its root. The figure code checks the window estimates against the seed-level differences and annual macro series.

## Build figure tables from evaluations

The following commands use the per-history folders written by `evaluation.run` and the output directories written by `evaluation.score`. The score directory for each command should contain that experiment's completed output. Each command writes to its own files inside the same figure-input directory.

```bash
python figures/prepare.py controls --histories runs --reference references/historical.csv --output figure-inputs
python figures/prepare.py historical --histories runs --age-reference references/age.csv --output figure-inputs
python figures/prepare.py criteria --scores scores/historical --output figure-inputs
python figures/prepare.py acceptance --scores scores/acceptance --output figure-inputs
python figures/prepare.py spatial --scores scores/spatial --output figure-inputs
python figures/prepare.py forward --scores scores/forward --histories runs --reference references/forward.csv --output figure-inputs
python -m evaluation.score tariff runs/tariff figure-inputs/tariff --exposures path/to/exposures.csv
python gtha_labor_marl/curriculum.py --training training-runs --output figure-inputs/training/training_diagnostics_source.csv
```

The controls step reads the native annual age-cohort arrays, monthly endpoint flows, wage moments, recruitment counters and market-close capacity. It pairs each control with the same trained history before summarising annual fit. It covers the three policy configurations at five seeds, fixed firms at five seeds and the reservation-wage intervention at two seeds. The historical step uses both market sizes, reconstructs resident and cohort-based rates separately, and computes elapsed durations and age-specific rates. The age reference can be omitted for analyses other than ED2. The spatial step uses the reported within-history salary quantiles and the distinct market and post-renewal vacancy measurements. The forward step subtracts the same calendar month in 2019 within policy and pairs the on/off arms. Supplying its histories also builds the transition and duration tables for ED7 and S8.

| Supplied empirical reference | Required fields and units |
| --- | --- |
| Historical regional rates | `year,employment_rate,participation_rate,unemployment_rate`, annual shares for 2006–2016 |
| Age-specific regional rates | `year,group,metric,value_pct`, annual percentages for ages 15–24, 25–54, 55–64 and 65+, using `EPOP`, `LFPR` and `ULF`. Use ASCII hyphens in the group codes. The age-65+ unemployment rate is outside the displayed comparison. |
| Forward regional rates | `year,month,employment_rate,participation_rate,unemployment_rate,temp_layoff_share_population,marginal_share_population`, monthly shares for 2017–2023. The temporary-layoff and marginal-attachment series end in 2021. Leave subsequent observations blank. |

## Framework map

Fig. 1 combines the supplied vector diagram with the three selected metropolitan boundaries. To create its map input from Statistics Canada's 2006 census metropolitan area and census agglomeration digital boundary file, place the geographic NAD83 shapefile and its `.dbf`, `.shx` and `.prj` companions together. Create the `spatial` subdirectory, then run the command with the shapefile path. [Statistics Canada boundary reference guide](https://publications.gc.ca/Collection/Statcan/92-196-X/92-196-XIE2006001.pdf)

```bash
python figures/census_map.py path/to/gcma000a06a_e.shp figure-inputs/spatial/census_2006_cma_locator.json
python figures/render.py --inputs figure-inputs --output figures-out --figure Fig1
```
