"""Smoke test: confirm the freshly re-split MICA III donor domains load cleanly on the HPC.

Run this in the same environment that will actually use these domains downstream
(e.g. nhood_cluster.py) — on the HPC:

    source ~/Projects/AirScape_analysis/muspan/bin/activate
    python test_load_split_mica_domains.py

Loads each of the six per-donor domains one at a time, catching and reporting any error per
file instead of stopping at the first failure, so you get a full pass/fail summary in one run.
"""

import traceback
from pathlib import Path

import numpy as np

import muspan as ms

base_dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)
domains_dir = base_dir / "output" / "muspan" / "domains"

DOMAIN_NAMES = [
    "MICA_III_311",
    "MICA_III_315",
    "MICA_III_319",
    "MICA_III_325",
    "MICA_III_337",
    "MICA_III_379",
]

print(f"NumPy version in this environment: {np.__version__}")
print(f"Looking for domains in: {domains_dir}")
print()

results = {}

for name in DOMAIN_NAMES:
    domain_path = domains_dir / f"{name}_muspan_domain.muspan"
    print(f"--- {name} ---")
    print(f"path: {domain_path}")
    print(f"exists: {domain_path.exists()}")

    if not domain_path.exists():
        results[name] = "MISSING"
        print("SKIPPED — file not found\n")
        continue

    try:
        domain = ms.io.load_domain(str(domain_path))
        print(
            domain
        )  # muspan's own summary — n_objects, collections, labels, networks, etc.
        results[name] = "OK"
        print("LOADED OK\n")
        del domain
    except Exception:
        results[name] = "FAILED"
        print("FAILED TO LOAD:")
        traceback.print_exc()
        print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, status in results.items():
    print(f"{status:8s} {name}")

n_ok = sum(1 for s in results.values() if s == "OK")
n_total = len(DOMAIN_NAMES)
print()
print(f"{n_ok}/{n_total} domains loaded successfully.")

if n_ok < n_total:
    raise SystemExit(1)
