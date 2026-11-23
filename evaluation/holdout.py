import math
from collections import Counter, defaultdict
import csv
import io
from pathlib import Path
from statistics import fmean
import zipfile

METRICS = ("employment_rate", "participation_rate", "unemployment_rate")
ARMS = ("forward_FN_OFF", "forward_FN_ON", "forward_RZ_OFF", "forward_RZ_ON")
SEEDS = tuple(range(42, 47))
MONTHS = tuple(((year, month) for year in range(2017, 2020) for month in range(1, 13)))


def require(test, message):
    if not test:
        raise ValueError(message)


def check_rates(actual, expected, label):
    require(
        len(actual) == len(expected)
        and all((math.isclose(a, b, rel_tol=0, abs_tol=1e-11) for a, b in zip(actual, expected))),
        f"Score or rate mismatch at {label}",
    )


def rates(employed, unemployed, population):
    return (
        employed / population,
        (employed + unemployed) / population,
        unemployed / (employed + unemployed),
    )


def december(path):
    totals = Counter()
    with zipfile.ZipFile(path) as archive, archive.open("pub1216.csv") as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
        for row in reader:
            require(
                (int(row["SURVYEAR"]), int(row["SURVMNTH"])) == (2016, 12), "Wrong survey month"
            )
            if int(row["PROV"]) != 35 or row["CMA"].strip().lstrip("0") not in {"4", "5"}:
                continue
            status, weight = (int(row["LFSSTAT"]), float(row["FINALWT"]))
            if status in {1, 2, 3, 4} and weight > 0:
                require(1 <= int(row["AGE_12"]) <= 12, "Invalid adult age code")
                totals[status] += weight
    employed, unemployed, population = (totals[1] + totals[2], totals[3], sum(totals.values()))
    return rates(employed, unemployed, population)


def observed(path):
    data = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["year"]), int(row["month"]))
            require(key not in data, "Duplicate survey month")
            data[key] = rates(
                *(float(row[f"weighted_{k}"]) for k in ("employed", "unemployed", "population"))
            )
            check_rates(
                data[key],
                [float(row[k]) for k in ("employment_share_population", *METRICS[1:])],
                key,
            )
    require(set(data) == set(MONTHS), "Wrong survey window")
    return data


def ensemble(monthly_path):
    groups = defaultdict(dict)
    with Path(monthly_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            arm = row["case"]
            if arm not in ARMS or row["year"] not in {"2017", "2018", "2019"}:
                continue
            month = (int(row["year"]), int(row["month"]))
            seed, index = (int(row["policy_seed"]), int(row["replication_index"]))
            key = (arm, month, seed)
            require(
                row["metric_operator"] == "native_TH_535_537_adult_LFS_after_market",
                "Unexpected all-worker operator",
            )
            require(
                index not in groups[key] and int(row["evaluation_seed"]) == 100 * seed + index,
                "Invalid or duplicate history identity",
            )
            data = rates(*(int(row["lfs_" + k]) for k in ("employed", "unemployed", "population")))
            check_rates(data, [float(row[k]) for k in METRICS], key)
            groups[key][index] = data
    first = {}
    for month in MONTHS:
        for arm in ARMS:
            policy_means = []
            for seed in SEEDS:
                group = groups.pop((arm, month, seed))
                require(group.keys() == set(range(50)), "Incomplete policy")
                policy_means.append(tuple(map(fmean, zip(*group.values()))))
            result = tuple(map(fmean, zip(*policy_means)))
            if arm == ARMS[0]:
                first[month] = result
            else:
                check_rates(result, first[month], (arm, month))
    require(not groups, "Extra policies or months")
    return first
