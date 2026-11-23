import torch
from math import radians, sin, cos, sqrt, asin
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)
import os

_USER_CACHE = Path(
    os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
)
CACHE_DIR = Path(os.environ.get("METRO_CACHE_DIR", _USER_CACHE / "metropolitan-labour-market"))


@dataclass
class TransitStation:
    name: str
    lat: float
    lon: float
    access_radius_km: float = 2.0


GO_LAKESHORE_WEST = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Exhibition", 43.6367, -79.4167),
    TransitStation("Mimico", 43.6153, -79.4944),
    TransitStation("Long Branch", 43.5931, -79.5422),
    TransitStation("Port Credit", 43.5506, -79.5858),
    TransitStation("Clarkson", 43.5117, -79.6206),
    TransitStation("Oakville", 43.4506, -79.6817),
    TransitStation("Bronte", 43.4278, -79.7083),
    TransitStation("Appleby", 43.3869, -79.7433),
    TransitStation("Burlington", 43.3447, -79.8019),
    TransitStation("Hamilton GO", 43.2533, -79.8694),
]
GO_LAKESHORE_EAST = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Danforth", 43.6847, -79.2931),
    TransitStation("Scarborough", 43.7139, -79.2561),
    TransitStation("Eglinton", 43.7231, -79.2344),
    TransitStation("Guildwood", 43.7539, -79.1939),
    TransitStation("Rouge Hill", 43.7772, -79.1622),
    TransitStation("Pickering", 43.8344, -79.0889),
    TransitStation("Ajax", 43.8503, -79.0356),
    TransitStation("Whitby", 43.8642, -78.9425),
    TransitStation("Oshawa", 43.8694, -78.8728),
]
GO_MILTON = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Kipling", 43.6372, -79.5364),
    TransitStation("Dixie", 43.5806, -79.5681),
    TransitStation("Cooksville", 43.5628, -79.5978),
    TransitStation("Erindale", 43.5428, -79.6361),
    TransitStation("Streetsville", 43.5725, -79.7128),
    TransitStation("Meadowvale", 43.5922, -79.7531),
    TransitStation("Lisgar", 43.5597, -79.7806),
    TransitStation("Milton", 43.5183, -79.8833),
]
GO_GEORGETOWN = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Bloor", 43.6606, -79.4217),
    TransitStation("Weston", 43.6986, -79.515),
    TransitStation("Etobicoke North", 43.7122, -79.5569),
    TransitStation("Malton", 43.7078, -79.635),
    TransitStation("Bramalea", 43.7028, -79.6892),
    TransitStation("Brampton", 43.6858, -79.7583),
    TransitStation("Mount Pleasant", 43.6858, -79.8278),
    TransitStation("Georgetown", 43.6539, -79.9208),
]
GO_RICHMOND_HILL = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Old Cummer", 43.7947, -79.3706),
    TransitStation("Langstaff", 43.8125, -79.4306),
    TransitStation("Richmond Hill", 43.8683, -79.4375),
]
GO_STOUFFVILLE = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Danforth", 43.6847, -79.2931),
    TransitStation("Scarborough", 43.7139, -79.2561),
    TransitStation("Agincourt", 43.7864, -79.2797),
    TransitStation("Milliken", 43.8167, -79.2806),
    TransitStation("Unionville", 43.8531, -79.3131),
    TransitStation("Centennial", 43.8636, -79.3369),
    TransitStation("Markham", 43.8728, -79.2611),
    TransitStation("Mount Joy", 43.8878, -79.2625),
    TransitStation("Stouffville", 43.9706, -79.2458),
]
GO_BARRIE = [
    TransitStation("Union", 43.6453, -79.3806),
    TransitStation("Downsview Park", 43.7539, -79.4783),
    TransitStation("Rutherford", 43.8361, -79.4889),
    TransitStation("Maple", 43.8614, -79.51),
    TransitStation("King City", 43.9222, -79.5283),
    TransitStation("Aurora", 44.0022, -79.4617),
    TransitStation("Newmarket", 44.0567, -79.4653),
    TransitStation("East Gwillimbury", 44.0978, -79.4628),
    TransitStation("Bradford", 44.1211, -79.5603),
    TransitStation("Barrie South", 44.3633, -79.705),
]
TTC_YUS = [
    TransitStation("Finch", 43.7808, -79.4147, 1.5),
    TransitStation("North York Centre", 43.7689, -79.4128, 1.5),
    TransitStation("Sheppard-Yonge", 43.7617, -79.4108, 1.5),
    TransitStation("York Mills", 43.7439, -79.4064, 1.5),
    TransitStation("Lawrence", 43.725, -79.4019, 1.5),
    TransitStation("Eglinton", 43.7058, -79.3981, 1.5),
    TransitStation("Davisville", 43.6972, -79.3961, 1.5),
    TransitStation("St. Clair", 43.6836, -79.3922, 1.5),
    TransitStation("Bloor-Yonge", 43.6711, -79.3858, 1.5),
    TransitStation("Wellesley", 43.6653, -79.3842, 1.5),
    TransitStation("College", 43.6614, -79.3831, 1.5),
    TransitStation("Dundas", 43.6564, -79.3803, 1.5),
    TransitStation("Queen", 43.6522, -79.3792, 1.5),
    TransitStation("King", 43.6489, -79.3781, 1.5),
    TransitStation("Union", 43.6453, -79.3806, 1.5),
    TransitStation("St. George", 43.6681, -79.3997, 1.5),
    TransitStation("Spadina Stn", 43.6672, -79.4039, 1.5),
    TransitStation("Dupont", 43.6744, -79.4069, 1.5),
    TransitStation("St. Clair West", 43.6839, -79.4147, 1.5),
    TransitStation("Eglinton West", 43.6992, -79.4353, 1.5),
    TransitStation("Glencairn", 43.7089, -79.4414, 1.5),
    TransitStation("Lawrence West", 43.7164, -79.4444, 1.5),
    TransitStation("Yorkdale", 43.7253, -79.4508, 1.5),
    TransitStation("Wilson", 43.7342, -79.45, 1.5),
    TransitStation("Downsview", 43.7536, -79.4789, 1.5),
]
TTC_BD = [
    TransitStation("Kipling Stn", 43.6369, -79.5361, 1.5),
    TransitStation("Islington", 43.6375, -79.5247, 1.5),
    TransitStation("Royal York", 43.6364, -79.5106, 1.5),
    TransitStation("Old Mill", 43.65, -79.4953, 1.5),
    TransitStation("Jane", 43.6497, -79.4839, 1.5),
    TransitStation("Runnymede", 43.6514, -79.4756, 1.5),
    TransitStation("High Park", 43.6539, -79.4667, 1.5),
    TransitStation("Keele", 43.6556, -79.4594, 1.5),
    TransitStation("Dundas West", 43.6569, -79.4528, 1.5),
    TransitStation("Lansdowne", 43.6594, -79.4428, 1.5),
    TransitStation("Dufferin", 43.6603, -79.4356, 1.5),
    TransitStation("Ossington", 43.6625, -79.4264, 1.5),
    TransitStation("Christie", 43.6642, -79.4186, 1.5),
    TransitStation("Bathurst", 43.6661, -79.4111, 1.5),
    TransitStation("Spadina BD", 43.6672, -79.4039, 1.5),
    TransitStation("St. George BD", 43.6681, -79.3997, 1.5),
    TransitStation("Bay", 43.6703, -79.3903, 1.5),
    TransitStation("Bloor-Yonge BD", 43.6711, -79.3858, 1.5),
    TransitStation("Sherbourne", 43.6722, -79.3764, 1.5),
    TransitStation("Castle Frank", 43.6736, -79.3686, 1.5),
    TransitStation("Broadview", 43.6769, -79.3586, 1.5),
    TransitStation("Chester", 43.6783, -79.3528, 1.5),
    TransitStation("Pape", 43.68, -79.345, 1.5),
    TransitStation("Donlands", 43.6806, -79.3378, 1.5),
    TransitStation("Greenwood", 43.6819, -79.3303, 1.5),
    TransitStation("Coxwell", 43.6839, -79.3228, 1.5),
    TransitStation("Woodbine", 43.6864, -79.3125, 1.5),
    TransitStation("Main Street", 43.6894, -79.3017, 1.5),
    TransitStation("Victoria Park", 43.695, -79.2894, 1.5),
    TransitStation("Warden", 43.7, -79.2794, 1.5),
    TransitStation("Kennedy", 43.7325, -79.2636, 1.5),
]
TTC_SRT = [
    TransitStation("Kennedy SRT", 43.7325, -79.2636, 1.5),
    TransitStation("Lawrence East", 43.7483, -79.2708, 1.5),
    TransitStation("Ellesmere", 43.7669, -79.2667, 1.5),
    TransitStation("Midland", 43.7703, -79.2717, 1.5),
    TransitStation("Scarborough Centre", 43.7742, -79.2578, 1.5),
    TransitStation("McCowan", 43.7758, -79.2489, 1.5),
]
ALL_GO_STATIONS: List[TransitStation] = (
    GO_LAKESHORE_WEST
    + GO_LAKESHORE_EAST
    + GO_MILTON
    + GO_GEORGETOWN
    + GO_RICHMOND_HILL
    + GO_STOUFFVILLE
    + GO_BARRIE
)
ALL_SUBWAY_STATIONS: List[TransitStation] = TTC_YUS + TTC_BD + TTC_SRT
SPEED_GO_EFFECTIVE = 65.0
SPEED_SUBWAY_EFFECTIVE = 30.0
SPEED_AUTO_URBAN = 25.0
SPEED_AUTO_MIXED = 32.0
SPEED_AUTO_HIGHWAY = 45.0
GO_ACCESS_TIME = 10.0
GO_EGRESS_TIME = 8.0
SUBWAY_ACCESS_TIME = 7.0
SUBWAY_EGRESS_TIME = 6.0
TRANSFER_PENALTY = 8.0
AUTO_OVERHEAD = 6.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def _auto_time(dist_km: float) -> float:
    if dist_km <= 15:
        return dist_km / SPEED_AUTO_URBAN * 60 + AUTO_OVERHEAD
    elif dist_km <= 30:
        t = 15 / SPEED_AUTO_URBAN * 60
        t += (dist_km - 15) / SPEED_AUTO_MIXED * 60
        return t + AUTO_OVERHEAD
    else:
        t = 15 / SPEED_AUTO_URBAN * 60
        t += 15 / SPEED_AUTO_MIXED * 60
        t += (dist_km - 30) / SPEED_AUTO_HIGHWAY * 60
        return t + AUTO_OVERHEAD


def _find_nearest_station(
    lat: float, lon: float, stations: List[TransitStation]
) -> Tuple[Optional[TransitStation], float]:
    best_station = None
    best_dist = float("inf")
    for station in stations:
        dist = haversine_km(lat, lon, station.lat, station.lon)
        if dist < station.access_radius_km and dist < best_dist:
            best_station = station
            best_dist = dist
    return (best_station, best_dist)


def compute_single_travel_time(
    origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float
) -> float:
    direct_dist = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    if direct_dist < 2.0:
        return max(5.0, direct_dist * 4.0)
    origin_go, _ = _find_nearest_station(origin_lat, origin_lon, ALL_GO_STATIONS)
    dest_go, _ = _find_nearest_station(dest_lat, dest_lon, ALL_GO_STATIONS)
    origin_sub, _ = _find_nearest_station(origin_lat, origin_lon, ALL_SUBWAY_STATIONS)
    dest_sub, _ = _find_nearest_station(dest_lat, dest_lon, ALL_SUBWAY_STATIONS)
    best = float("inf")
    if origin_go and dest_go:
        go_dist = haversine_km(origin_go.lat, origin_go.lon, dest_go.lat, dest_go.lon)
        t = GO_ACCESS_TIME + go_dist / SPEED_GO_EFFECTIVE * 60 + GO_EGRESS_TIME
        best = min(best, t)
    if origin_go and dest_sub and (not dest_go):
        go_to_union = haversine_km(origin_go.lat, origin_go.lon, 43.6453, -79.3806)
        union_to_dest = haversine_km(43.6453, -79.3806, dest_sub.lat, dest_sub.lon)
        t = (
            GO_ACCESS_TIME
            + go_to_union / SPEED_GO_EFFECTIVE * 60
            + TRANSFER_PENALTY
            + union_to_dest / SPEED_SUBWAY_EFFECTIVE * 60
            + SUBWAY_EGRESS_TIME
        )
        best = min(best, t)
    if origin_sub and dest_go and (not origin_go):
        sub_to_union = haversine_km(origin_sub.lat, origin_sub.lon, 43.6453, -79.3806)
        union_to_dest = haversine_km(43.6453, -79.3806, dest_go.lat, dest_go.lon)
        t = (
            SUBWAY_ACCESS_TIME
            + sub_to_union / SPEED_SUBWAY_EFFECTIVE * 60
            + TRANSFER_PENALTY
            + union_to_dest / SPEED_GO_EFFECTIVE * 60
            + GO_EGRESS_TIME
        )
        best = min(best, t)
    if origin_sub and dest_sub:
        sub_dist = haversine_km(origin_sub.lat, origin_sub.lon, dest_sub.lat, dest_sub.lon)
        t = SUBWAY_ACCESS_TIME + sub_dist / SPEED_SUBWAY_EFFECTIVE * 60 + SUBWAY_EGRESS_TIME
        best = min(best, t)
    best = min(best, _auto_time(direct_dist))
    return max(5.0, best)


class GTHATravelTimeCalculator:

    def __init__(
        self, da_centroids: Dict[int, Tuple[float, float]], device="cpu", use_cache: bool = True
    ):
        self.da_centroids = da_centroids
        self.device = device
        self.da_ids = sorted(da_centroids.keys())
        self.n_das = len(self.da_ids)
        self.da_id_to_idx = {da_id: idx for idx, da_id in enumerate(self.da_ids)}
        self.use_cache = use_cache

    def _get_cache_key(self) -> str:
        locations = [(str(da), tuple(map(float, self.da_centroids[da]))) for da in self.da_ids]
        return hashlib.sha256(repr(locations).encode()).hexdigest()[:16]

    def _get_cache_path(self) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = self._get_cache_key()
        return CACHE_DIR / f"travel_time_matrix_{self.n_das}das_{cache_key}.pt"

    def _load_from_cache(self) -> Optional[torch.Tensor]:
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                print(f"Loading travel time matrix from cache: {cache_path.name}")
                data = torch.load(cache_path, map_location="cpu", weights_only=True)
                matrix = data["matrix"]
                cached_n_das = data.get("n_das", matrix.shape[0])
                cached_da_ids = data.get("da_ids", [])
                if matrix.shape[0] == self.n_das and matrix.shape[1] == self.n_das:
                    if cached_da_ids and list(cached_da_ids) != self.da_ids:
                        print("Cached dissemination-area IDs differ, recomputing...")
                        return None
                    print(f"Loaded {self.n_das}x{self.n_das} travel time matrix from cache")
                    return matrix.to(self.device)
                else:
                    print(f"Cache size mismatch ({cached_n_das} vs {self.n_das}), recomputing...")
                    return None
            except Exception as e:
                print(f"Failed to load cache: {e}, recomputing...")
                return None
        return None

    def _save_to_cache(self, matrix: torch.Tensor):
        import os
        from uuid import uuid4

        cache_path = self._get_cache_path()
        temp = cache_path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            torch.save({"matrix": matrix.cpu(), "n_das": self.n_das, "da_ids": self.da_ids}, temp)
            os.replace(temp, cache_path)
            logger.info(f"Saved travel time matrix to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        finally:
            temp.unlink(missing_ok=True)

    def compute_travel_time_matrix(self) -> torch.Tensor:
        if self.use_cache:
            cached = self._load_from_cache()
            if cached is not None:
                return cached
        print(f"\n{'=' * 60}")
        print(f"Computing {self.n_das}x{self.n_das} corridor-aware travel time matrix...")
        print(
            f"Total pairs: {self.n_das * (self.n_das - 1) // 2:,}. The full-scale matrix takes 5 to 10 minutes."
        )
        print("The matrix is cached for later runs.")
        print(f"{'=' * 60}")
        import time
        import sys

        start_time = time.time()
        time_matrix = torch.zeros(self.n_das, self.n_das, dtype=torch.float32)
        total_pairs = self.n_das * (self.n_das - 1) // 2
        computed = 0
        log_interval = max(1, total_pairs // 20)
        for i in range(self.n_das):
            lat_i, lon_i = self.da_centroids[self.da_ids[i]]
            for j in range(i + 1, self.n_das):
                lat_j, lon_j = self.da_centroids[self.da_ids[j]]
                t = compute_single_travel_time(lat_i, lon_i, lat_j, lon_j)
                time_matrix[i, j] = t
                time_matrix[j, i] = t
                computed += 1
                if computed % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = computed / elapsed if elapsed > 0 else 0
                    remaining = (total_pairs - computed) / rate if rate > 0 else 0
                    pct = 100 * computed / total_pairs
                    print(
                        f"  Travel time matrix: {pct:5.1f}% | {elapsed:5.0f}s elapsed | ~{remaining:5.0f}s remaining"
                    )
                    sys.stdout.flush()
        time_matrix = time_matrix.to(self.device)
        elapsed = time.time() - start_time
        print(f"Travel time matrix computed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        try:
            upper_indices = torch.triu_indices(self.n_das, self.n_das, offset=1)
            upper_values = time_matrix[upper_indices[0], upper_indices[1]]
            if upper_values.numel() > 100000:
                sample_idx = torch.randperm(upper_values.numel())[:100000]
                sample = upper_values[sample_idx]
            else:
                sample = upper_values
            print(
                f"Travel time stats: mean={upper_values.mean():.1f}min, median={sample.median():.1f}min, p75={sample.quantile(0.75):.1f}min, max={upper_values.max():.1f}min"
            )
        except Exception as e:
            print(f"Could not compute stats: {e}")
        mem_mb = time_matrix.element_size() * time_matrix.numel() / (1024 * 1024)
        print(f"Matrix size: {self.n_das}x{self.n_das}, {mem_mb:.1f} MB")
        if self.use_cache:
            self._save_to_cache(time_matrix)
        return time_matrix
