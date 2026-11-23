import argparse
import hashlib
import json
import struct
from pathlib import Path

EXPECTED = {"35532": "Oshawa", "35535": "Toronto", "35537": "Hamilton"}


def extract(path):
    parts = {
        suffix: path.with_suffix(suffix).read_bytes() for suffix in (".dbf", ".shp", ".shx", ".prj")
    }
    if b"GCS_North_American_1983" not in parts[".prj"] or not parts[".prj"].startswith(b"GEOGCS"):
        raise ValueError("Expected NAD83 geographic coordinates")
    dbf, shapefile, index = (parts[s] for s in (".dbf", ".shp", ".shx"))
    count, header_size, row_size = struct.unpack_from("<IHH", dbf, 4)
    fields = {}
    start = 1
    for pos in range(32, header_size - 1, 32):
        name = dbf[pos : pos + 11].split(b"\x00")[0].decode("ascii")
        width = dbf[pos + 16]
        fields[name] = slice(start, start + width)
        start += width
    shapes = {}
    for row in range(count):
        record = dbf[header_size + row * row_size : header_size + (row + 1) * row_size]
        uid = record[fields["CMAUID"]].strip().decode("ascii")
        if uid not in EXPECTED:
            continue
        name = record[fields["CMANAME"]].strip().decode("ascii")
        assert name == EXPECTED[uid]
        offset = 8 + 2 * struct.unpack_from(">i", index, 100 + 8 * row)[0]
        assert struct.unpack_from("<i", shapefile, offset)[0] == 5
        nrings, ncoords = struct.unpack_from("<ii", shapefile, offset + 36)
        assert nrings == 1
        points = [
            tuple((round(v, 6) for v in struct.unpack_from("<dd", shapefile, offset + 48 + 16 * i)))
            for i in range(ncoords)
        ]
        assert points[0] == points[-1]
        shapes[uid] = {"name": name, "ring_lon_lat": points}
    assert shapes.keys() == EXPECTED.keys()
    return {
        "source": "Statistics Canada, 2006 Census metropolitan area and census agglomeration digital boundary file, gcma000a06a_e",
        "catalogue": "https://publications.gc.ca/Collection/Statcan/92-196-X/92-196-XIE2006001.pdf",
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in parts.items()},
        "coordinates": "NAD83 longitude and latitude (degrees, six decimals)",
        "areas": shapes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(extract(args.shapefile), separators=(",", ":")), encoding="utf-8"
    )
