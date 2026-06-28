"""
scraper_pipeline_v4.py
=======================
Option A: Selenium-based scraper for all three JS-rendered sources.

Sources:
  - SCFI          : TradingEconomics (Selenium/Highcharts) — same as your scrape.py
  - PMI Manuf.    : TradingEconomics (tedata package)
  - PMI Services  : TradingEconomics (tedata package)
  - Trade Volume  : MacroMicro (Selenium/XHR intercept)
  - Brent Crude   : FRED API (no change)
  - CNY/USD       : FRED API (no change)

Requirements:
    pip install requests pandas fredapi beautifulsoup4 supabase openpyxl selenium webdriver-manager tedata

Run:
    python scraper_pipeline_v4.py
"""

import os, io, re, time, logging, requests, pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("scraper")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "1a27e76845c52829f17c5c57caf0f122")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://otdzxhuqgsgtnxuptzwk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_f53alv4kF8JtH5z80eg-vA_bRh05zux")
TABLE        = "scfi_weekly"

# Path to your chromedriver — update if needed
CHROMEDRIVER_PATH = os.environ.get(
    "CHROMEDRIVER_PATH",
    r"C:\Users\User\Desktop\scraper\chromedriver.exe"
)

def to_weekly_friday(series: pd.Series) -> pd.Series:
    series = series.dropna().copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.resample("W-FRI").last().ffill()


def make_driver(headless: bool = True):
    """Create a Chrome WebDriver, auto-downloading chromedriver if needed."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # Try local chromedriver first, then auto-download
    try:
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception:
        pass

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        raise RuntimeError(f"Could not create Chrome WebDriver: {e}")


# ── 1. SCFI — TradingEconomics Highcharts (same as your scrape.py) ────────────
def fetch_scfi_selenium() -> pd.Series:
    log.info("Fetching SCFI (Selenium/TradingEconomics)…")
    driver = None
    try:
        driver = make_driver(headless=True)
        driver.get("https://tradingeconomics.com/commodity/containerized-freight-index")
        time.sleep(8)

        raw = driver.execute_script("""
            try {
                var chart = Highcharts.charts.filter(c => c)[0];
                return chart.series[0].options.data;
            } catch(e) { return null; }
        """)

        if not raw:
            log.warning("  SCFI: Highcharts not found, trying alternate extraction")
            raw = driver.execute_script("""
                try {
                    var charts = Highcharts.charts.filter(c => c);
                    for (var i = 0; i < charts.length; i++) {
                        var s = charts[i].series[0].options.data;
                        if (s && s.length > 10) return s;
                    }
                } catch(e) {}
                return null;
            """)

        if raw and len(raw) > 0:
            records = []
            for pt in raw:
                try:
                    if isinstance(pt, list):
                        date = pd.to_datetime(pt[0], unit="ms") if isinstance(pt[0], (int, float)) else pd.to_datetime(pt[0])
                        val  = float(pt[1])
                    elif isinstance(pt, dict):
                        date = pd.to_datetime(pt.get("date") or pt.get("x"))
                        val  = float(pt.get("y") or pt.get("value") or pt.get("close"))
                    else:
                        continue
                    records.append((date, val))
                except (TypeError, ValueError, KeyError):
                    continue

            if records:
                s = pd.Series(dict(records)).sort_index()
                log.info(f"  SCFI: {len(s)} rows, latest {s.index.max().date()} = {s.iloc[-1]:.2f}")
                return to_weekly_friday(s)

        log.warning("  SCFI: could not extract from Highcharts")
        return pd.Series(dtype=float)

    except Exception as e:
        log.error(f"  SCFI Selenium failed: {e}")
        return pd.Series(dtype=float)
    finally:
        if driver:
            driver.quit()


# ── 2 & 3. PMI — TradingEconomics via Selenium ───────────────────────────────
def _fetch_te_series(url: str, label: str) -> pd.Series:
    """Fetch a TradingEconomics series via Selenium.

    tedata is not usable in this environment — it can't obtain a driver for
    either firefox or chrome (its own driver resolution, separate from
    make_driver()/CHROMEDRIVER_PATH, fails outright) — so we skip it and go
    straight to Selenium.
    """
    log.info(f"Fetching {label} (Selenium)…")
    return _fetch_te_selenium(url, label)


def _fetch_te_selenium(url: str, label: str, max_retries: int = 3) -> pd.Series:
    """Fallback: use raw Selenium + Highcharts for TradingEconomics.

    TE appears to rate-limit / reset the connection when hit again too soon
    after a previous scrape in the same run (e.g. right after SCFI). Retry
    with an increasing cooldown and a fresh driver on each attempt.
    """
    for attempt in range(1, max_retries + 1):
        driver = None
        try:
            if attempt > 1:
                cooldown = 15 * attempt
                log.info(f"  {label}: retry {attempt}/{max_retries} after {cooldown}s cooldown…")
                time.sleep(cooldown)

            driver = make_driver(headless=True)
            driver.get(url)
            time.sleep(8)

            raw = driver.execute_script("""
                try {
                    var charts = Highcharts.charts.filter(c => c);
                    for (var i = 0; i < charts.length; i++) {
                        var s = charts[i].series[0].options.data;
                        if (s && s.length > 5) return s;
                    }
                } catch(e) {}
                return null;
            """)

            if raw:
                records = []
                for pt in raw:
                    try:
                        if isinstance(pt, list):
                            date = pd.to_datetime(pt[0], unit="ms") if isinstance(pt[0], (int, float)) else pd.to_datetime(pt[0])
                            val  = float(pt[1])
                        elif isinstance(pt, dict):
                            date = pd.to_datetime(pt.get("date") or pt.get("x"))
                            val  = float(pt.get("y") or pt.get("value"))
                        else:
                            continue
                        records.append((date, val))
                    except Exception:
                        continue
                if records:
                    s = pd.Series(dict(records)).sort_index()
                    log.info(f"  {label} (Selenium): {len(s)} rows, latest {s.index.max().date()} = {s.iloc[-1]:.2f}")
                    return to_weekly_friday(s)

            log.warning(f"  {label}: Selenium returned no usable data (attempt {attempt}/{max_retries})")

        except Exception as e:
            log.warning(f"  {label} Selenium failed (attempt {attempt}/{max_retries}): {e}")
        finally:
            if driver:
                driver.quit()

    log.error(f"  {label}: all Selenium attempts exhausted")
    return pd.Series(dtype=float)


def fetch_pmi_manufacturing() -> pd.Series:
    return _fetch_te_series(
        "https://tradingeconomics.com/united-states/business-confidence",
        "PMI Manufacturing"
    )

def fetch_pmi_services() -> pd.Series:
    log.info("Cooling down 15s before next TradingEconomics request…")
    time.sleep(15)
    return _fetch_te_series(
        "https://tradingeconomics.com/united-states/non-manufacturing-pmi",
        "PMI Services"
    )


# ── 4. Trade Volume — CPB World Trade Monitor (direct xlsx download) ─────────
def fetch_trade_volume_cpb(months_back: int = 4):
    """
    CPB (Netherlands Bureau for Economic Policy Analysis) publishes the World
    Trade Monitor monthly as a free, public .xlsx — no auth, no Selenium.
    MacroMicro's "Global Trade Volume" chart is just a paywalled re-chart of
    this same CPB series, which is why we couldn't get real data out of it
    (the actual chart-data endpoint is gated; only a view-tracking endpoint
    was reachable). CPB is the original source, so we go straight there.

    URL pattern: https://www.cpb.nl/system/files/cpbmedia/
                 cpb-world-trade-monitor-{month_name}-{year}.xlsx
    e.g. cpb-world-trade-monitor-january-2026.xlsx

    CPB has a ~2 month publication lag, so we try the most recent few months
    going backwards until one resolves.

    Also computes YoY % change, matching MacroMicro's companion
    "World - World Trade Volume (YoY)" series (CPB source, Month freq,
    Percent units) — confirmed via MacroMicro's own series metadata and
    independently corroborated by Saxo's CPB commentary ("minus 0.8%
    year-on-year in July"): this is a standard 12-month-back calendar
    comparison on the monthly index, i.e.
        YoY(t) = (value[t] - value[t-12 months]) / value[t-12 months] * 100
    computed on the monthly series BEFORE resampling to weekly, so the
    comparison stays calendar-month-accurate rather than drifting from a
    52-week rolling window on forward-filled weekly data.

    Returns (level_weekly, yoy_weekly) — either may be an empty Series if
    unavailable.
    """
    log.info("Fetching Trade Volume (CPB World Trade Monitor)…")

    today = datetime.utcnow()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(months_back):
        # walk back i months from today
        year = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year -= 1
        month_name = datetime(year, month, 1).strftime("%B").lower()
        url = f"https://www.cpb.nl/system/files/cpbmedia/cpb-world-trade-monitor-{month_name}-{year}.xlsx"

        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200 or len(resp.content) < 1000:
                log.info(f"  CPB: {month_name} {year} not available (status {resp.status_code})")
                continue

            log.info(f"  CPB: found {month_name} {year} file ({len(resp.content)} bytes)")
            xls = pd.ExcelFile(io.BytesIO(resp.content))
            log.info(f"  CPB: sheets = {xls.sheet_names}")

            monthly_series = _parse_cpb_workbook(xls)
            if monthly_series is not None and not monthly_series.empty:
                log.info(f"  Trade Volume: {len(monthly_series)} rows, "
                         f"latest {monthly_series.index.max().date()} = {monthly_series.iloc[-1]:.2f}")

                # YoY on the monthly series, calendar-month-accurate
                yoy_monthly = monthly_series.pct_change(periods=12) * 100
                yoy_monthly = yoy_monthly.dropna()
                if not yoy_monthly.empty:
                    log.info(f"  Trade Volume YoY: {len(yoy_monthly)} rows, "
                             f"latest {yoy_monthly.index.max().date()} = {yoy_monthly.iloc[-1]:.2f}%")
                else:
                    log.warning("  Trade Volume YoY: not enough history (<12 months) to compute")

                level_weekly = to_weekly_friday(monthly_series)
                yoy_weekly = to_weekly_friday(yoy_monthly) if not yoy_monthly.empty else pd.Series(dtype=float)
                return level_weekly, yoy_weekly
            else:
                log.warning(f"  CPB: downloaded {month_name} {year} but could not locate trade volume series in it")
                _debug_dump_cpb_workbook(xls)

        except Exception as e:
            log.warning(f"  CPB: {month_name} {year} fetch/parse failed: {e}")
            continue

    log.error("  Trade Volume: no CPB file could be downloaded/parsed in the last "
              f"{months_back} months")
    return pd.Series(dtype=float), pd.Series(dtype=float)


def _debug_dump_cpb_workbook(xls: pd.ExcelFile):
    """Print sheet names and first few rows of each so we can see the real
    layout if _parse_cpb_workbook fails to find the series automatically."""
    for sheet_name in xls.sheet_names:
        try:
            raw = xls.parse(sheet_name, header=None, nrows=8)
            log.info(f"  CPB debug — sheet '{sheet_name}' shape={raw.shape}")
            for r in range(min(8, len(raw))):
                row_preview = [str(x)[:30] for x in raw.iloc[r].tolist()[:8]]
                log.info(f"    row {r}: {row_preview}")
        except Exception as e:
            log.info(f"  CPB debug — sheet '{sheet_name}' failed to preview: {e}")


def _parse_cpb_workbook(xls: pd.ExcelFile) -> pd.Series:
    """
    CPB's actual layout (confirmed from a real download) is wide, not tall:

      row 3:  ['...', '...', '...', '...', '...', '2000m01', '2000m02', '2000m03', ...]
      row 7:  ['...', 'World trade', 'tgz_w1_qnmi_sn', <2021 value>, nan, 52.59, 52.91, 53.07, ...]

    i.e. one header row holds month codes like "2000m01" starting at some
    column, and each data row holds a label (e.g. "World trade") plus a
    numeric series running across the same columns. We want the 'trade_out'
    sheet's row labeled "World trade" (the headline world merchandise trade
    volume index), not 'inpro_out' (industrial production).
    """
    target_sheet = None
    for name in xls.sheet_names:
        if "trade" in name.lower():
            target_sheet = name
            break
    if target_sheet is None:
        target_sheet = xls.sheet_names[0]

    raw = xls.parse(target_sheet, header=None)

    # Find the header row containing month codes like "2000m01"
    month_code_re = re.compile(r"^\d{4}m\d{2}$")
    header_row_idx = None
    date_cols = None
    for r in range(min(10, len(raw))):
        row = raw.iloc[r].astype(str)
        matches = row.str.match(month_code_re, na=False)
        if matches.sum() > 10:
            header_row_idx = r
            date_cols = raw.columns[matches]
            break

    if header_row_idx is None:
        return None

    # Parse month codes -> real dates
    month_codes = raw.iloc[header_row_idx, date_cols].astype(str).tolist()
    dates = []
    for code in month_codes:
        m = re.match(r"(\d{4})m(\d{2})", code)
        if m:
            dates.append(pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1))
        else:
            dates.append(pd.NaT)

    # Find the "World trade" row (the headline index, not a regional breakdown)
    label_col = 1  # observed: labels sit in column index 1
    best_row = None
    for r in range(header_row_idx + 1, len(raw)):
        label = str(raw.iloc[r, label_col]).strip().lower()
        if label == "world trade":
            best_row = r
            break
    if best_row is None:
        # fallback: first row whose label contains "world" and "trade"
        for r in range(header_row_idx + 1, len(raw)):
            label = str(raw.iloc[r, label_col]).strip().lower()
            if "world" in label and "trade" in label:
                best_row = r
                break
    if best_row is None:
        return None

    values = pd.to_numeric(raw.iloc[best_row, date_cols], errors="coerce")
    s = pd.Series(values.values, index=pd.DatetimeIndex(dates))
    s = s[~s.index.isna()].dropna().sort_index()
    return s if len(s) > 10 else None


# ── 5 & 6. Brent + CNY/USD — FRED (unchanged) ────────────────────────────────
def fetch_brent(fred: Fred) -> pd.Series:
    log.info("Fetching Brent Crude (FRED)…")
    try:
        s = fred.get_series("DCOILBRENTEU").dropna()
        log.info(f"  Brent: {len(s)} rows, latest {s.index.max().date()} = {s.iloc[-1]:.2f}")
        return to_weekly_friday(s)
    except Exception as e:
        log.error(f"  Brent failed: {e}")
        return pd.Series(dtype=float)

def fetch_cny_usd(fred: Fred) -> pd.Series:
    log.info("Fetching CNY/USD (FRED)…")
    try:
        s = fred.get_series("DEXCHUS").dropna()
        log.info(f"  CNY/USD: {len(s)} rows, latest {s.index.max().date()} = {s.iloc[-1]:.4f}")
        return to_weekly_friday(s)
    except Exception as e:
        log.error(f"  CNY/USD failed: {e}")
        return pd.Series(dtype=float)


# ── Supabase fallback for any empty series ────────────────────────────────────
def supabase_fallback(col: str) -> pd.Series:
    """Read an existing column from Supabase to forward-fill gaps."""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = client.table(TABLE).select(f"date,{col}").order("date", desc=False).execute()
        if result.data:
            records = {pd.to_datetime(r["date"]): r[col] for r in result.data if r.get(col) is not None}
            if records:
                return to_weekly_friday(pd.Series(records))
    except Exception:
        pass
    return pd.Series(dtype=float)


# ── Merge & Upsert ────────────────────────────────────────────────────────────
def merge_and_upsert(series_dict: dict):
    log.info("Merging…")
    today = datetime.utcnow().replace(tzinfo=None)
    spine = pd.date_range(start="2020-01-03", end=today + timedelta(days=7), freq="W-FRI")
    df = pd.DataFrame(index=spine)
    df.index.name = "date"

    col_map = {
        "scfi":              "scfi",
        "brent":             "brent_crude",
        "cny_usd":           "cny_usd",
        "pmi_manufacturing": "pmi_manufacturing",
        "pmi_services":      "pmi_services",
        "trade_volume":      "trade_volume_index",
        "trade_volume_yoy":  "trade_volume_yoy",
    }

    for key, series in series_dict.items():
        col = col_map.get(key, key)
        if series is None or series.empty:
            log.warning(f"  {key} empty — loading from Supabase")
            series = supabase_fallback(col)
        if not series.empty:
            series.index = pd.to_datetime(series.index).tz_localize(None)
            df[col] = series.reindex(df.index, method="ffill")

    if "scfi" not in df.columns or df["scfi"].isna().all():
        log.error("SCFI empty — aborting")
        return None

    df = df.dropna(subset=["scfi"]).reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    log.info(f"  {len(df)} rows, latest: {df['date'].max()}")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    records = df.to_dict(orient="records")
    # df.where(pd.notnull(df), None) doesn't reliably scrub every NaN across
    # mixed dtypes, and a bare float('nan') breaks Supabase's JSON encoder
    # ("Out of range float values are not JSON compliant: nan"). Sanitize
    # every value explicitly instead.
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and pd.isna(v):
                rec[k] = None
    for i in range(0, len(records), 500):
        client.table(TABLE).upsert(records[i:i+500], on_conflict="date").execute()
        log.info(f"  → rows {i+1}–{min(i+500, len(records))}")

    log.info("✅ Upsert complete")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("FreightAI Scraper v4 (Selenium)")
    log.info(f"Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    fred = Fred(api_key=FRED_API_KEY)

    scfi_series = fetch_scfi_selenium()

    # Cooldown before hitting tradingeconomics.com again — back-to-back
    # Selenium sessions against the same domain trigger ERR_CONNECTION_RESET.
    log.info("Cooling down 20s before next TradingEconomics request…")
    time.sleep(20)

    trade_volume_level, trade_volume_yoy = fetch_trade_volume_cpb()

    series = {
        "scfi":              scfi_series,
        "brent":             fetch_brent(fred),
        "cny_usd":           fetch_cny_usd(fred),
        "pmi_manufacturing": fetch_pmi_manufacturing(),
        "pmi_services":      fetch_pmi_services(),
        "trade_volume":      trade_volume_level,
        "trade_volume_yoy":  trade_volume_yoy,
    }

    df = merge_and_upsert(series)
    if df is None:
        return

    log.info("── Latest values ─────────────────────────────────")
    latest = df[df["date"] == df["date"].max()].iloc[0]
    for col in ["scfi", "brent_crude", "cny_usd", "pmi_manufacturing", "pmi_services",
                "trade_volume_index", "trade_volume_yoy"]:
        val = latest.get(col)
        status = f"{val:.4f}" if val is not None and val == val else "N/A"
        log.info(f"  {col:25s}: {status}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()