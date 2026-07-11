"""
One-off migration: collapse scfi_weekly's duplicate Monday/Friday rows
(root cause: a historical CSV backfill via supabase_upload.py wrote
Monday-dated rows, while scraper_pipeline.py's ongoing weekly-Friday
spine writes Friday-dated rows for the same calendar weeks - Postgres's
on_conflict="date" upsert never caught these as duplicates since they
have different date keys).

Policy:
  - Week has both a Monday row and a Friday row -> delete the Monday
    row, keep the Friday row untouched (it's already correctly dated
    and is the actively-maintained pipeline's convention).
  - Week has only a Monday row (no Friday counterpart) -> re-date it
    onto that ISO week's Friday, preserving all its data.
  - Week has only a Friday row -> leave alone.

Run manually: python _dedupe_scfi_weekly.py [--dry-run]
"""
import sys
import datetime
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
from supabase import create_client

DRY_RUN = "--dry-run" in sys.argv

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

res = client.table("scfi_weekly").select("*").order("date").execute()
df = pd.DataFrame(res.data)
df["date"] = pd.to_datetime(df["date"])
df["iso_year"] = df["date"].dt.isocalendar().year
df["iso_week"] = df["date"].dt.isocalendar().week
df["weekday"] = df["date"].dt.day_name()

# ── 1. Backup ────────────────────────────────────────────────────────────
backup_path = f"data_backups/scfi_weekly_backup_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
df.drop(columns=["iso_year", "iso_week", "weekday"]).to_csv(backup_path, index=False)
print(f"Backed up {len(df)} rows -> {backup_path}")

# ── 2. Compute the plan ─────────────────────────────────────────────────
to_delete = []   # list of date strings (Monday rows in a duplicate week)
to_redate = []   # list of (old_date_str, new_date_str)

for (iso_year, iso_week), g in df.groupby(["iso_year", "iso_week"]):
    if len(g) == 1:
        row = g.iloc[0]
        if row["weekday"] != "Friday":
            # lone Monday row, no Friday counterpart - re-date to that week's Friday
            friday = pd.Timestamp.fromisocalendar(int(iso_year), int(iso_week), 5)
            to_redate.append((row["date"].strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")))
        continue

    fridays = g[g["weekday"] == "Friday"]
    mondays = g[g["weekday"] != "Friday"]
    if len(fridays) >= 1:
        # keep Friday row(s) as-is, delete every non-Friday row in this week
        for _, row in mondays.iterrows():
            to_delete.append(row["date"].strftime("%Y-%m-%d"))
    else:
        # no Friday row at all in this "duplicate" group (shouldn't happen given
        # the audit found only Mon/Fri weekdays, but handle defensively):
        # keep the first row, re-date it to the week's Friday, delete the rest
        keep = g.iloc[0]
        friday = pd.Timestamp.fromisocalendar(int(iso_year), int(iso_week), 5)
        to_redate.append((keep["date"].strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")))
        for _, row in g.iloc[1:].iterrows():
            to_delete.append(row["date"].strftime("%Y-%m-%d"))

print(f"Plan: delete {len(to_delete)} rows, re-date {len(to_redate)} rows")
print(f"Expected final row count: {len(df) - len(to_delete)}")

if DRY_RUN:
    print("\n--dry-run set, stopping before any writes.")
    print("Sample deletes:", to_delete[:5])
    print("Sample redates:", to_redate[:5])
    sys.exit(0)

# ── 3. Execute ───────────────────────────────────────────────────────────
deleted = 0
for d in to_delete:
    client.table("scfi_weekly").delete().eq("date", d).execute()
    deleted += 1
print(f"Deleted {deleted} rows.")

redated = 0
skipped_collisions = 0
for old_date, new_date in to_redate:
    # defensive: skip if something already occupies the target date
    existing = client.table("scfi_weekly").select("date").eq("date", new_date).execute()
    if existing.data:
        skipped_collisions += 1
        print(f"  SKIP re-date {old_date} -> {new_date}: target date already occupied")
        continue
    client.table("scfi_weekly").update({"date": new_date}).eq("date", old_date).execute()
    redated += 1
print(f"Re-dated {redated} rows ({skipped_collisions} skipped due to collision).")

final = client.table("scfi_weekly").select("date", count="exact").execute()
print(f"Final row count: {final.count}")
