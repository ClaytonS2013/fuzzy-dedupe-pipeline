"""
main.py
=======

This script orchestrates a nightly deduplication job for veterinary practice
records stored in Supabase.  It is designed to run on a serverless runner
such as Railway, where it can be scheduled via a cron expression.  The
engine pulls all raw records from a source table, performs fuzzy matching
and clustering using `pandas` and `fuzzywuzzy`, and writes the deduped
results back to a separate results table.  It also optionally logs
metadata about each run to a log table.

The behaviour of the script is governed via environment variables so that
the same code can be reused in different environments without modification.

Required environment variables
-----------------------------

```
SUPABASE_URL           # e.g. https://myproj.supabase.co
SUPABASE_SERVICE_KEY   # service role key with read/write permissions
SOURCE_TABLE           # table to read raw records from (default: practice_records)
RESULTS_TABLE          # table to write deduped results (default: dedupe_results)
LOG_TABLE              # table to log run metadata (default: dedupe_log)
THRESHOLD              # fuzziness threshold for names (default: 90)
BATCH_SIZE             # fetch batch size for Supabase paging (default: 5000)
```

To run the script locally you can create a `.env` file with the above
variables and `pip install -r requirements.txt`.  On Railway, set these
variables in the project settings and configure a scheduled trigger (e.g.
`0 3 * * *` for 3 AM UTC).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import requests
from fuzzywuzzy import fuzz

# Flag to enable local smoke test mode. When the SMOKE_TEST environment
# variable is set to "1", the pipeline skips Supabase calls, seeds a
# synthetic dataset, and runs the deduplication logic locally. This
# allows quick validation of the core logic without hitting external
# dependencies. See the `seed_synthetic_rows` function below for the
# sample data.
SMOKE: bool = os.getenv("SMOKE_TEST", "0") == "1"



def get_env(name: str, default: str | None = None) -> str:
    """Helper to fetch an environment variable with an optional default.

    If the variable is missing and no default is provided the process
    terminates with an error.  Supplying a default allows optional
    configuration values.
    """
    value = os.getenv(name)
    if value is None:
        if default is None:
            # Print to stderr so logs capture the missing env var
            print(f"Error: missing required environment variable {name}", file=sys.stderr)
            sys.exit(1)
        return default
    return value


def seed_synthetic_rows(n: int = 5) -> List[Dict[str, object]]:
    """Generate a small synthetic dataset for smoke testing.

    Parameters
    ----------
    n : int, optional
        Number of rows to return. Defaults to 5. If `n` is
        less than the length of the base dataset, the list will be
        truncated; if greater, rows will be repeated until the
        requested size is reached.

    Returns
    -------
    List[Dict[str, object]]
        A list of dictionaries representing veterinary practices with
        slight variations to test the deduplication logic.
    """
    base = [
        {
            "practice_name": "Happy Paws Vet Clinic",
            "address": "123 Main St",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
        },
        {
            "practice_name": "Happy Pawz Veterinary Clinic",
            "address": "123 Main Street",
            "city": "Austin",
            "state": "TX",
            "zip": "78701",
        },
        {
            "practice_name": "Oak Hills Animal Hosp.",
            "address": "88 Oak Hills Rd",
            "city": "Dallas",
            "state": "TX",
            "zip": "75201",
        },
        {
            "practice_name": "Oak Hills Animal Hospital",
            "address": "88 Oak Hills Road",
            "city": "Dallas",
            "state": "TX",
            "zip": "75201",
        },
        {
            "practice_name": "Sunset Vet Center",
            "address": "999 5th Ave",
            "city": "Houston",
            "state": "TX",
            "zip": "77001",
        },
    ]
    rows: List[Dict[str, object]] = []
    while len(rows) < n:
        rows.extend(base)
    return rows[:n]


def fetch_supabase_table(api_url: str, api_key: str, table: str, batch_size: int) -> List[Dict[str, object]]:
    """
    Fetch all rows from a Supabase table using range pagination.

    `batch_size` controls the number of rows retrieved per request; increasing
    it reduces the number of network round trips at the cost of larger responses.
    """
    base_url = f"{api_url}/rest/v1/{table}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    all_rows: List[Dict[str, object]] = []
    range_start = 0
    while True:
        range_end = range_start + batch_size - 1
        range_header = {"Range": f"{range_start}-{range_end}"}
        response = requests.get(base_url, headers={**headers, **range_header})
        if response.status_code not in (200, 206):
            raise RuntimeError(
                f"Failed to fetch rows: HTTP {response.status_code} – {response.text}"
            )
        rows = response.json()
        if not rows:
            break
        all_rows.extend(rows)
        # If fewer rows than requested were returned then we've reached the end
        if len(rows) < batch_size:
            break
        range_start += batch_size
    return all_rows


def upsert_supabase_rows(api_url: str, api_key: str, table: str, rows: Sequence[Dict[str, object]]):
    """
    Upsert a list of dictionaries into a Supabase table.

    The table should define a primary key or unique constraint for idempotent behaviour.
    This function returns the JSON response from Supabase if available.
    """
    if not rows:
        return None
    url = f"{api_url}/rest/v1/{table}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    response = requests.post(url, headers=headers, data=json.dumps(rows))
    if response.status_code not in (201, 200, 204):
        raise RuntimeError(
            f"Failed to upsert rows: HTTP {response.status_code} – {response.text}"
        )
    return response.json() if response.content else None


def block_dataframe(df: pd.DataFrame, column: str) -> Dict[str, List[int]]:
    """
    Simple blocking on the first character of `column`.  This reduces the
    comparison space by grouping records that start with the same letter.
    """
    blocks: Dict[str, List[int]] = {}
    # Fill missing values with empty strings and lower-case everything
    for idx, value in df[column].fillna("").str.lower().items():
        key = value[0] if value else "_blank_"
        blocks.setdefault(key, []).append(idx)
    return blocks


def fuzzy_cluster(block_indices: List[int], df: pd.DataFrame, name_thresh: int) -> List[List[int]]:
    """
    Greedy clustering within a block.

    A new cluster is started by the first unseen record; any later record with
    name and address similarity above thresholds is grouped with the seed.  The
    address threshold is set slightly lower to allow minor differences (street
    vs st).  Adjust `name_thresh` via the THRESHOLD env var.
    """
    used: set[int] = set()
    clusters: List[List[int]] = []
    # Derive a secondary threshold for addresses
    addr_thresh = max(name_thresh - 5, 80)
    for i in block_indices:
        if i in used:
            continue
        seed = df.loc[i]
        cluster = [i]
        used.add(i)
        for j in block_indices:
            if j <= i or j in used:
                continue
            other = df.loc[j]
            # Convert to strings to avoid TypeError if values are not strings
            name_score = fuzz.token_sort_ratio(str(seed.get('practice_name', '')), str(other.get('practice_name', '')))
            addr_score = fuzz.token_sort_ratio(str(seed.get('address', '')), str(other.get('address', '')))
            if name_score >= name_thresh and addr_score >= addr_thresh:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    return clusters


def pick_representative(cluster: List[int], df: pd.DataFrame) -> Dict[str, object]:
    """
    Create a canonical record from a cluster by starting with the first
    record and filling in missing values from subsequent records.  If you
    have additional logic (e.g. choosing the most recently updated record),
    modify this function accordingly.
    """
    primary = df.loc[cluster[0]].to_dict()
    for idx in cluster[1:]:
        rec = df.loc[idx]
        for col, value in rec.items():
            if pd.isna(primary.get(col)) or primary.get(col) == "":
                primary[col] = value
    return primary


def dedupe_records(df: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Execute blocking and fuzzy clustering across the entire DataFrame.

    Returns a deduped DataFrame and a Series mapping each original index to a
    cluster identifier.  The cluster mapping can be persisted for auditing.
    """
    blocks = block_dataframe(df, column="practice_name")
    clusters: List[List[int]] = []
    for indices in blocks.values():
        clusters.extend(fuzzy_cluster(indices, df, threshold))
    cluster_ids: Dict[int, int] = {}
    for cid, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            cluster_ids[idx] = cid
    cleaned_records: List[Dict[str, object]] = []
    for cluster in clusters:
        cleaned_records.append(pick_representative(cluster, df))
    cleaned_df = pd.DataFrame(cleaned_records)
    cluster_series = pd.Series(cluster_ids)
    return cleaned_df, cluster_series


def log_run(api_url: str, api_key: str, table: str, *, start_time: float, num_raw: int, num_clusters: int, num_clean: int) -> None:
    """Persist a simple log record of this run to the specified Supabase table."""
    record = {
        "run_timestamp": datetime.utcfromtimestamp(start_time).isoformat() + "Z",
        "records_processed": num_raw,
        "clusters": num_clusters,
        "records_cleaned": num_clean,
    }
    upsert_supabase_rows(api_url, api_key, table, [record])


def main() -> None:
    """Entrypoint for the deduplication job."""
    start_ts = time.time()
    # Read threshold and batch size first; these apply in both smoke and production modes.
    threshold = int(get_env("THRESHOLD", "90"))
    batch_size = int(get_env("BATCH_SIZE", "5000"))

    # If SMOKE is enabled, perform the smoke test before fetching any Supabase
    # configuration. This avoids requiring SUPABASE_URL and other variables.
    if SMOKE:
        print("[SMOKE] Running local smoke test — skipping Supabase calls", flush=True)
        synthetic_rows = seed_synthetic_rows()
        df = pd.DataFrame(synthetic_rows)
        print(f"[SMOKE] Seeded {len(df)} synthetic rows", flush=True)
        cleaned_df, cluster_series = dedupe_records(df, threshold=threshold)
        print(
            f"[SMOKE] Identified {cluster_series.nunique()} clusters; writing {len(cleaned_df)} unique records",
            flush=True,
        )
        print("PIPELINE Complete", flush=True)
        return

    # In production mode, fetch required Supabase environment variables.
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    source_table = get_env("SOURCE_TABLE", "practice_records")
    results_table = get_env("RESULTS_TABLE", "dedupe_results")
    log_table = get_env("LOG_TABLE", "dedupe_log")

    # Print configuration for observability
    print(
        f"\n>>> Starting deduplication job at {datetime.utcnow().isoformat()}Z",
        flush=True,
    )
    print(
        f"Configuration: source={source_table}, results={results_table}, log={log_table}, threshold={threshold}, batch_size={batch_size}",
        flush=True,
    )


    # Pull raw records from Supabase
    rows = fetch_supabase_table(supabase_url, supabase_key, source_table, batch_size)
    if not rows:
        print("No records found; job exiting.", flush=True)
        return
    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} records from Supabase", flush=True)

    # Deduplicate
    cleaned_df, cluster_series = dedupe_records(df, threshold=threshold)
    print(
        f"Identified {cluster_series.nunique()} clusters; writing {len(cleaned_df)} unique records",
        flush=True,
    )

    # Remove 'address' column before writing results if present
    cleaned_df = cleaned_df.drop(columns=["address", "city"], errors="ignore")
    records_to_write = cleaned_df.to_dict(orient="records")
    upsert_supabase_rows(supabase_url, supabase_key, results_table, records_to_write)
    print(f"Uploaded deduped records to {results_table}", flush=True)

    # Log the run
    try:
        log_run(
            supabase_url,
            supabase_key,
            log_table,
            start_time=start_ts,
            num_raw=len(df),
            num_clusters=cluster_series.nunique(),
            num_clean=len(cleaned_df),
        )
        print(f"Logged run details to {log_table}", flush=True)
    except Exception as e:
        print(f"Warning: failed to log run details: {e}", file=sys.stderr)

    print(
        f"Deduplication job complete at {datetime.utcnow().isoformat()}Z",
        flush=True,
    )


if __name__ == "__main__":
    main()
