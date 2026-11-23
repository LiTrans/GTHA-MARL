# Evaluate the labour-market model

Run these commands from the repository root after generating the synthetic population and training a policy. Use a separate output directory for every history. Each history runs on the CPU. The population design seed is 42. The simulation seed is 100 times the policy seed plus the history index. The demonstration checkpoint was trained with 250 residents and 80 firm slots, so the commands below pass `--workers 250 --firms 80`. For a checkpoint from the full schedule pass 1000 and 80, or omit both.

```sh
python -m evaluation.run historical --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --months 12 --output demo/history42
```

`history.json` identifies the completed history and its checkpoint. `simulation_log.json` contains the annual cohort series, matching funnel and mechanism counts. `monthly.json` contains opening-to-closing worker transitions, closing stocks by age, unemployment duration and wages. Its `market_capacity` and `renewal_capacity` entries record job capacity before and after demographic renewal. `monthly.csv` provides the three headline rates for plotting. All rates in this CSV are fractions.

## Rates and scoring populations

| Measure | Population, timing and denominator |
| --- | --- |
| Historical employment, participation and unemployment | Residents of the three simulated census metropolitan areas after each market step. The adult population comprises employed, unemployed by labour-market status and outside the labour force. Employment divides employment by population. Participation divides employed plus status-unemployed by population. Unemployment divides status-unemployed by that labour force. See `monthly.json → months → age_resident_raw_and_aligned → all` and `write_monthly` in `run.py`. Historical annual rates aggregate person-month counts before division. |
| Active-search unemployment | `search_unemployment_rate` in `monthly.csv` divides searching unemployed residents by employed plus searching unemployed residents. The companion participation rate uses that labour force. |
| Four historical criteria | The scorer uses the simulated non-immigrant annual series in `simulation_log.json → cohort_macros_by_age_native → annual`. It computes search-aligned unemployment across ages 15–99, and the share outside the labour force for ages 35–44, 55–64 and 65–99. The full window is 2006–2016 and the terminal window is 2014–2016. `gtha_labor_marl/windowed_metrics.py` defines the operators. `gtha_labor_marl/eval_aggregation.py` defines the bands and how to classify uncertainty intervals. |
| Prime-age job finding | In each history, sum the same-person transitions to employment among workers unemployed by status at market opening and aged 25–54, then divide by the summed opening count. Use February–December in 2006–2011 and include every route into employment. Average the resulting ratios equally over 50 histories per policy. |
| Holdout and pandemic | Use Toronto and Hamilton, ages 15 and older, at market close. The survey-aligned measure includes temporary layoffs and other searching unemployed workers. Oshawa contributes to the historical three-area resident rates. The forward survey comparison uses Toronto and Hamilton. |

## Historical designs

For a full panel, train one policy for each seed from 42 through 46. Evaluate each policy using history indices 0 through 49, the same initial population specification and matched simulation seeds. Place the resulting directories under a common input directory. The full historical horizon is 132 months. A short demonstration run exercises the code but cannot receive a historical verdict.

| Family in `evaluation.run` | Checkpoint | Design |
| --- | --- | --- |
| `historical` | Final Stage 3 | Trained workers and firms. The study uses 1,000/80 and 2,500/200 initial residents/firm slots. |
| `untrained_random` | `stage1/initial_checkpoint.pt` | Both policies use their untrained parameters. |
| `untrained_heuristic` | `stage1/initial_checkpoint.pt` | Untrained workers and the fixed firm action. |
| `firm_fixed` | Final Stage 3 | Trained workers paired with the fixed firm rule. |
| `bundle_off` | Final Stage 3 | Disable the human-capital and wage-history bundle for evaluation. The study comparison uses seeds 42 and 43. |
| `spatial` | Final Stage 3 | Passive commute, vacancy and wage observers at both historical scales. |
| `worker_groups` | Final Stage 3 | One year with the trained policies. Adds `worker_groups.json` with employment, unemployment, wage and tenure for each of the nine worker groups. |

Pass the size explicitly with `--workers` and `--firms`. The worker and firm input pools must cover the requested sizes. Keep the checkpoint beside its training-stage `manifest.json`. The historical and spatial scale transfer multiplies both training counts by 2.5. The fixed-firm and wage-history interventions record their specific parameter changes in each completed history. `score historical` summarises each complete seed block and compares available matched seeds to the trained 1,000/80 condition. It also reports four historical criteria for each seed. `score spatial` requires both scales. The spatial analysis assigns travel times to observed job locations and measures home-area access to month-end vacancies within 60 minutes. It weights vacancy-access observations by unemployed residents with known travel support, including months with zero vacancies.

```sh
python -m evaluation.run worker_groups --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --output demo/worker-groups42
python -m evaluation.score historical runs historical-scores
python -m evaluation.score spatial runs spatial-scores
```

## Offer acceptance

Run `acceptance --arm baseline` first. These histories cover 2006–2019. Use `score acceptance-fit` when all 250 baseline histories exist. For each policy it fits one intercept to the actual 2006–2016 baseline offers in Toronto, Hamilton and Oshawa. Reuse that policy's fit when running the paired `recentred` arm. In that arm the four offer coefficients for wage, skill match, commute and job security are zero. The scorer checks realised offer acceptance in Toronto and Hamilton and estimates the changes in wage, match quality, commute at hire and holdout macro error. Its five-policy Student intervals use one paired estimate per policy. It applies Holm adjustment across the four outcome tests.

```sh
python -m evaluation.run acceptance --arm baseline --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --output runs/acceptance/baseline/42/0
python -m evaluation.score acceptance-fit runs/acceptance fits
python -m evaluation.run acceptance --arm recentred --fit fits/policy_42.json --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --output runs/acceptance/recentred/42/0
python -m evaluation.score acceptance runs/acceptance acceptance-scores --fit-directory fits --targets holdout.csv
```

The illustrative run commands specify the first matched history. Supply all histories before the panel scoring commands. The holdout CSV supplies `year,month,employment_share_population,participation_rate,unemployment_rate` for the 36 months in 2017–2019, with rates in fractions. Use a survey extract available to you or fictional targets for a demonstration.

## Pandemic, tariff and entrant analyses

Generate illustrative schedules, or supply compatible driver files. The generated schedules have invented magnitudes for testing. The pandemic file covers 18 industries, and the tariff exposure file links industry rules to product groups and input baskets.

```sh
python -m evaluation.scenarios demo/scenarios
python -m evaluation.run forward --arm RZ_ON --covid-driver demo/scenarios/covid.csv --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --output runs/forward/RZ_ON/42/0
python -m evaluation.run tariff --arm policy_record_persistent --covid-driver demo/scenarios/covid.csv --tariff-driver demo/scenarios/tariff.csv --exposures demo/scenarios/exposures.csv --data synthetic_data/example --checkpoint demo/policy42/stage3/final_checkpoint.pt --seed 42 --history 0 --workers 250 --firms 80 --output runs/tariff/persistent/42/0
```

The four forward arms are `FN_OFF`, `FN_ON`, `RZ_OFF` and `RZ_ON`. FN holds the standard separation drivers at 2019 levels. RZ uses realised drivers. ON and OFF control the pandemic regime. Each forward history covers 2006–2023. The forward scorer produces seed-level series, five-seed means and matched ON-minus-OFF differences. To score survey fit, supply 84 rows for 2017–2023 with `year,month,employment_rate,participation_rate,unemployment_rate,temp_layoff_share_population,marginal_share_population`, all in fractions. April 2020 is excluded from the pandemic error calculation. Layoff and marginal attachment use changes from the corresponding calendar month in 2019. For the December persistence benchmark also supply the Statistics Canada December 2016 public-use ZIP containing `pub1216.csv`.

The entrant scorer uses the 250 realised-driver, pandemic-on histories. It links workers through the July renewal boundary, compares opening and market-close statuses and reconciles each stock with the model's Toronto–Hamilton observation. The 2017–2019 breakdown reports each entry group's share of unemployed residents and contribution to the unemployment rate. The scorer follows the 2007–2018 July entrants to first observed employment at months 1, 3, 6 and 12 across the three simulated areas. It reports registry exits before first employment. The same worker records give consecutive-month job-finding rates for ages 16–69 for January–November origins in 2006–2011 and trailing six-month recession rates at ages 15 and older. `consecutive_month_job_finding_policy.csv` reports both the pooled-exposure rate and the equal-history sensitivity. The survey CSV may leave the two joblessness stock outcomes blank outside their 2019 reference and 2020–2021 scoring months.

Tariff histories cover 2006–2030. Supply the eight scenarios in `evaluation/tariff.py → SCENARIOS`. Scoring requires 50 matched histories per scenario and policy, totalling 2,000. Thirteen paired contrasts report windowed outcomes with seed-level uncertainty and explicit geographic, product-group and industry coverage.

```sh
python -m evaluation.score forward runs/forward forward-scores --targets forward.csv --december-lfs december_2016.zip
python -m evaluation.score entrants runs/forward entrant-scores
python -m evaluation.score tariff runs/tariff tariff-scores --exposures demo/scenarios/exposures.csv
```
