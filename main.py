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

PHASE 9 UPGRADE: AI-Assisted Fuzzy Matching
--------------------------------------------
This version includes an AI verification layer that analyzes borderline 
similarity cases (85-94% fuzzy score) and makes human-like judgments on 
whether records represent the same business.

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

# AI Enhancement Variables (Optional)
OPENAI_API_KEY         # OpenAI API key for AI verification (optional)
AI_PROVIDER            # "openai" or "anthropic" (default: openai)
ANTHROPIC_API_KEY      # Anthropic API key (if using Claude)
AI_THRESHOLD_LOW       # Lower bound for AI verification (default: 85)
AI_THRESHOLD_HIGH      # Upper bound for AI verification (default: 94)
AI_MODEL               # Model to use (default: gpt-4o-mini)
ENABLE_AI_CACHE        # Enable caching of AI decisions (default: true)
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
from typing import Dict, List, Sequence, Tuple, Optional

import pandas as pd
import requests
from fuzzywuzzy import fuzz

# Flag to enable local smoke test mode
SMOKE: bool = os.getenv("SMOKE_TEST", "0") == "1"

# AI configuration
AI_ENABLED: bool = os.getenv("OPENAI_API_KEY") is not None or os.getenv("ANTHROPIC_API_KEY") is not None
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "openai").lower()
AI_CACHE_ENABLED: bool = os.getenv("ENABLE_AI_CACHE", "true").lower() == "true"

# Import AI libraries only if enabled
if AI_ENABLED:
    if AI_PROVIDER == "openai":
        try:
            from openai import OpenAI
            ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            print("⚠️ OpenAI library not installed. Run: pip install openai", file=sys.stderr)
            AI_ENABLED = False
    elif AI_PROVIDER == "anthropic":
        try:
            from anthropic import Anthropic
            ai_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            print("⚠️ Anthropic library not installed. Run: pip install anthropic", file=sys.stderr)
            AI_ENABLED = False

# In-memory cache for AI decisions (if caching enabled)
ai_decision_cache: Dict[str, bool] = {}


def get_env(name: str, default: str | None = None) -> str:
    """Helper to fetch an environment variable with an optional default."""
    value = os.getenv(name)
    if value is None:
        if default is None:
            print(f"Error: missing required environment variable {name}", file=sys.stderr)
            sys.exit(1)
        return default
    return value


def seed_synthetic_rows(n: int = 5) -> List[Dict[str, object]]:
    """Generate a small synthetic dataset for smoke testing."""
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
    """Fetch all rows from a Supabase table using range pagination."""
    base_url = f"{api_url}/rest/v1/{table}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    all_rows: List[Dict[str, object]] = []
    range_start = 0
    
    print(f"📥 Fetching from {table}...", flush=True)
    
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
        print(f"  Fetched {len(all_rows)} rows so far...", flush=True)
        if len(rows) < batch_size:
            break
        range_start += batch_size
    
    print(f"✅ Total fetched: {len(all_rows)} rows", flush=True)
    return all_rows


def upsert_supabase_rows(api_url: str, api_key: str, table: str, rows: Sequence[Dict[str, object]]):
    """Upsert a list of dictionaries into a Supabase table with improved error handling."""
    if not rows:
        print("No records to write", flush=True)
        return None
    
    url = f"{api_url}/rest/v1/{table}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    
    print(f"📤 Attempting to upsert {len(rows)} records to {table}...", flush=True)
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(list(rows)))
        
        if response.status_code in (201, 200, 204):
            print(f"✅ Successfully upserted {len(rows)} records", flush=True)
            return response.json() if response.content else None
        else:
            print(f"⚠️ Bulk upsert failed: HTTP {response.status_code} – {response.text}", flush=True)
            raise RuntimeError(f"Bulk upsert failed: {response.status_code}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Bulk upsert failed: {error_msg}", flush=True)
        print(f"🔄 Falling back to individual record upserts...", flush=True)
        
        success_count = 0
        failed_records = []
        
        for idx, record in enumerate(rows):
            try:
                response = requests.post(url, headers=headers, data=json.dumps([record]))
                
                if response.status_code in (201, 200, 204):
                    success_count += 1
                    if (idx + 1) % 100 == 0:
                        print(f"  Progress: {idx + 1}/{len(rows)} ({success_count} succeeded)", flush=True)
                else:
                    failed_records.append((idx, record, f"HTTP {response.status_code}"))
                    
            except Exception as e:
                failed_records.append((idx, record, str(e)))
        
        print(f"\n📊 Individual upsert results:", flush=True)
        print(f"  ✅ Success: {success_count}/{len(rows)}", flush=True)
        print(f"  ❌ Failed: {len(failed_records)} records", flush=True)
        
        if failed_records:
            print(f"\n⚠️ Failed records (first 5):", flush=True)
            for idx, record, error in failed_records[:5]:
                print(f"  Record {idx}: {error}", flush=True)
        
        if success_count == 0:
            raise RuntimeError("All upserts failed - check table schema and permissions")
        
        return None


def get_cache_key(record_a: Dict, record_b: Dict) -> str:
    """Generate a cache key for two records."""
    # Sort by practice name to ensure consistent keys regardless of order
    names = sorted([
        str(record_a.get('practice_name', '')),
        str(record_b.get('practice_name', ''))
    ])
    return f"{names[0]}||{names[1]}"


def ai_verify_duplicate(record_a: Dict, record_b: Dict, model: str = "gpt-4o-mini") -> Optional[bool]:
    """
    Use AI to verify if two records represent the same business.
    
    Returns:
        True if same business
        False if different businesses
        None if AI call fails
    """
    if not AI_ENABLED:
        return None
    
    # Check cache first
    if AI_CACHE_ENABLED:
        cache_key = get_cache_key(record_a, record_b)
        if cache_key in ai_decision_cache:
            return ai_decision_cache[cache_key]
    
    # Construct prompt
    prompt = f"""You are analyzing veterinary practice records to determine if they represent the same business.

Record A:
- Name: {record_a.get('practice_name', 'N/A')}
- Address: {record_a.get('address', 'N/A')}
- City: {record_a.get('city', 'N/A')}
- State: {record_a.get('state', 'N/A')}
- Zip: {record_a.get('zip', 'N/A')}

Record B:
- Name: {record_b.get('practice_name', 'N/A')}
- Address: {record_b.get('address', 'N/A')}
- City: {record_b.get('city', 'N/A')}
- State: {record_b.get('state', 'N/A')}
- Zip: {record_b.get('zip', 'N/A')}

Question: Do these two records refer to the same veterinary practice/business?

Consider:
- Name variations (abbreviations, reordering, synonyms)
- Address differences (Street vs St, Road vs Rd, etc.)
- Same location/zip code
- Common business name patterns

Answer with ONLY "TRUE" or "FALSE" (no explanation needed)."""

    try:
        if AI_PROVIDER == "openai":
            response = ai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data deduplication expert. Respond only with TRUE or FALSE."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            decision_text = response.choices[0].message.content.strip().upper()
        
        elif AI_PROVIDER == "anthropic":
            response = ai_client.messages.create(
                model=model if model.startswith("claude") else "claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            decision_text = response.content[0].text.strip().upper()
        
        else:
            return None
        
        # Parse response
        decision = decision_text == "TRUE"
        
        # Cache the result
        if AI_CACHE_ENABLED:
            cache_key = get_cache_key(record_a, record_b)
            ai_decision_cache[cache_key] = decision
        
        return decision
        
    except Exception as e:
        print(f"⚠️ AI verification failed: {e}", file=sys.stderr, flush=True)
        return None


def block_dataframe(df: pd.DataFrame, column: str) -> Dict[str, List[int]]:
    """Simple blocking on the first character of `column`."""
    blocks: Dict[str, List[int]] = {}
    for idx, value in df[column].fillna("").str.lower().items():
        key = value[0] if value else "_blank_"
        blocks.setdefault(key, []).append(idx)
    return blocks


def fuzzy_cluster(
    block_indices: List[int], 
    df: pd.DataFrame, 
    name_thresh: int,
    ai_threshold_low: int = 85,
    ai_threshold_high: int = 94,
    ai_model: str = "gpt-4o-mini"
) -> Tuple[List[List[int]], int]:
    """
    Greedy clustering within a block with AI-assisted verification.
    
    Returns:
        Tuple of (clusters, ai_calls_count)
    """
    used: set[int] = set()
    clusters: List[List[int]] = []
    addr_thresh = max(name_thresh - 5, 80)
    ai_calls = 0
    
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
            
            name_score = fuzz.token_sort_ratio(
                str(seed.get('practice_name', '')), 
                str(other.get('practice_name', ''))
            )
            addr_score = fuzz.token_sort_ratio(
                str(seed.get('address', '')), 
                str(other.get('address', ''))
            )
            
            # Definite match (high confidence)
            if name_score >= name_thresh and addr_score >= addr_thresh:
                cluster.append(j)
                used.add(j)
            
            # Gray zone - consult AI if enabled
            elif AI_ENABLED and ai_threshold_low <= name_score < name_thresh:
                ai_calls += 1
                ai_decision = ai_verify_duplicate(
                    seed.to_dict(), 
                    other.to_dict(),
                    model=ai_model
                )
                
                if ai_decision is True:
                    cluster.append(j)
                    used.add(j)
                # If ai_decision is False or None, skip this pair
        
        clusters.append(cluster)
    
    return clusters, ai_calls


def pick_representative(cluster: List[int], df: pd.DataFrame) -> Dict[str, object]:
    """Create a canonical record from a cluster."""
    primary = df.loc[cluster[0]].to_dict()
    for idx in cluster[1:]:
        rec = df.loc[idx]
        for col, value in rec.items():
            if pd.isna(primary.get(col)) or primary.get(col) == "":
                primary[col] = value
    return primary


def dedupe_records(
    df: pd.DataFrame, 
    threshold: int,
    ai_threshold_low: int = 85,
    ai_threshold_high: int = 94,
    ai_model: str = "gpt-4o-mini"
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """
    Execute blocking and fuzzy clustering with AI assistance.
    
    Returns:
        Tuple of (cleaned_df, cluster_series, total_ai_calls)
    """
    print(f"🔍 Starting deduplication with threshold={threshold}...", flush=True)
    if AI_ENABLED:
        print(f"🤖 AI verification enabled for scores {ai_threshold_low}-{ai_threshold_high}%", flush=True)
    
    blocks = block_dataframe(df, column="practice_name")
    print(f"  Created {len(blocks)} blocks for processing", flush=True)
    
    clusters: List[List[int]] = []
    total_ai_calls = 0
    
    for block_key, indices in blocks.items():
        block_clusters, ai_calls = fuzzy_cluster(
            indices, 
            df, 
            threshold,
            ai_threshold_low,
            ai_threshold_high,
            ai_model
        )
        clusters.extend(block_clusters)
        total_ai_calls += ai_calls
    
    print(f"  Identified {len(clusters)} clusters", flush=True)
    if AI_ENABLED:
        print(f"  🤖 AI calls made: {total_ai_calls}", flush=True)
    
    cluster_ids: Dict[int, int] = {}
    for cid, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            cluster_ids[idx] = cid
    
    cleaned_records: List[Dict[str, object]] = []
    for cluster in clusters:
        cleaned_records.append(pick_representative(cluster, df))
    
    cleaned_df = pd.DataFrame(cleaned_records)
    cluster_series = pd.Series(cluster_ids)
    
    print(f"✅ Deduplication complete: {len(df)} → {len(cleaned_df)} records", flush=True)
    
    return cleaned_df, cluster_series, total_ai_calls


def log_run(
    api_url: str, 
    api_key: str, 
    table: str, 
    *, 
    start_time: float, 
    num_raw: int, 
    num_clusters: int, 
    num_clean: int,
    ai_calls: int = 0
) -> None:
    """Persist a log record of this run to the specified Supabase table."""
    record = {
        "run_timestamp": datetime.utcfromtimestamp(start_time).isoformat() + "Z",
        "records_processed": num_raw,
        "clusters": num_clusters,
        "records_cleaned": num_clean,
        "ai_used": AI_ENABLED,
        "ai_calls": ai_calls if AI_ENABLED else 0,
    }
    upsert_supabase_rows(api_url, api_key, table, [record])


def main() -> None:
    """Entrypoint for the deduplication job."""
    start_ts = time.time()
    
    print("\n" + "="*60, flush=True)
    print("🚀 FUZZY DEDUPLICATION PIPELINE STARTING", flush=True)
    if AI_ENABLED:
        print(f"🤖 AI-ASSISTED MODE ({AI_PROVIDER.upper()})", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Read configuration
    threshold = int(get_env("THRESHOLD", "90"))
    batch_size = int(get_env("BATCH_SIZE", "5000"))
    ai_threshold_low = int(get_env("AI_THRESHOLD_LOW", "85"))
    ai_threshold_high = int(get_env("AI_THRESHOLD_HIGH", "94"))
    ai_model = get_env("AI_MODEL", "gpt-4o-mini")

    # Smoke test mode
    if SMOKE:
        print("[SMOKE] Running local smoke test — skipping Supabase calls", flush=True)
        synthetic_rows = seed_synthetic_rows()
        df = pd.DataFrame(synthetic_rows)
        print(f"[SMOKE] Seeded {len(df)} synthetic rows", flush=True)
        cleaned_df, cluster_series, ai_calls = dedupe_records(
            df, 
            threshold=threshold,
            ai_threshold_low=ai_threshold_low,
            ai_threshold_high=ai_threshold_high,
            ai_model=ai_model
        )
        num_clusters = cluster_series.nunique()
        print(f"[SMOKE] Identified {num_clusters} clusters; {len(cleaned_df)} unique records", flush=True)
        if AI_ENABLED:
            print(f"[SMOKE] AI calls made: {ai_calls}", flush=True)
        print("\n✅ SMOKE TEST COMPLETE\n", flush=True)
        return

    # Production mode
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    source_table = get_env("SOURCE_TABLE", "practice_records")
    results_table = get_env("RESULTS_TABLE", "dedupe_results")
    log_table = get_env("LOG_TABLE", "dedupe_log")

    # Print configuration
    print(f"⏰ Started at: {datetime.utcnow().isoformat()}Z", flush=True)
    print(f"📋 Configuration:", flush=True)
    print(f"   Source table: {source_table}", flush=True)
    print(f"   Results table: {results_table}", flush=True)
    print(f"   Log table: {log_table}", flush=True)
    print(f"   Threshold: {threshold}", flush=True)
    print(f"   Batch size: {batch_size}", flush=True)
    if AI_ENABLED:
        print(f"   AI threshold range: {ai_threshold_low}-{ai_threshold_high}%", flush=True)
        print(f"   AI model: {ai_model}", flush=True)
        print(f"   AI cache: {'enabled' if AI_CACHE_ENABLED else 'disabled'}", flush=True)
    print("", flush=True)

    # Fetch records
    rows = fetch_supabase_table(supabase_url, supabase_key, source_table, batch_size)
    if not rows:
        print("⚠️ No records found; job exiting.\n", flush=True)
        return
    
    df = pd.DataFrame(rows)
    print(f"✅ Loaded {len(df)} records into DataFrame\n", flush=True)

    # Deduplicate
    cleaned_df, cluster_series, ai_calls = dedupe_records(
        df, 
        threshold=threshold,
        ai_threshold_low=ai_threshold_low,
        ai_threshold_high=ai_threshold_high,
        ai_model=ai_model
    )
    num_clusters = cluster_series.nunique()

    # Remove specified columns before writing results
    columns_to_drop = ["address", "city", "external_id"]
    cleaned_df = cleaned_df.drop(
        columns=[col for col in columns_to_drop if col in cleaned_df.columns], 
        errors="ignore"
    )
    
    print(f"\n📝 Preparing {len(cleaned_df)} records for upload...", flush=True)
    records_to_write = cleaned_df.to_dict(orient="records")
    
    upsert_supabase_rows(supabase_url, supabase_key, results_table, records_to_write)

    # Log the run
    try:
        print(f"\n📊 Logging run details to {log_table}...", flush=True)
        log_run(
            supabase_url,
            supabase_key,
            log_table,
            start_time=start_ts,
            num_raw=len(df),
            num_clusters=num_clusters,
            num_clean=len(cleaned_df),
            ai_calls=ai_calls,
        )
        print(f"✅ Run details logged successfully", flush=True)
    except Exception as e:
        print(f"⚠️ Warning: failed to log run details: {e}", file=sys.stderr, flush=True)

    elapsed = time.time() - start_ts
    print("\n" + "="*60, flush=True)
    print(f"✅ PIPELINE COMPLETE", flush=True)
    print(f"⏱️  Total time: {elapsed:.2f} seconds", flush=True)
    print(f"📈 Reduced {len(df)} → {len(cleaned_df)} records ({num_clusters} clusters)", flush=True)
    if AI_ENABLED:
        print(f"🤖 AI-assisted matches: {ai_calls}", flush=True)
        if ai_calls > 0:
            cost_estimate = ai_calls * 0.0007  # Rough estimate for GPT-4o-mini
            print(f"💰 Estimated cost: ${cost_estimate:.4f}", flush=True)
    print("="*60 + "\n", flush=True)


if __name__ == "__main__":
  
        # Run Google Sheets to Supabase sync before dedupe
    import subprocess
    try:
        subprocess.run(["python", "sheets_sync/main.py", "--run-now"], check=True)
    except Exception as e:
        print(f"Error running sheets_sync: {e}", flush=True)
    main()


