"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles of its
Exophase earn rate at the case's target, from the house's reads of the trophy page.

Method (per trophy, in logit space):
  1. Take the distinct values of the trophy's earn rate over the case's history
     (Exophase refreshes in steps, so consecutive identical reads carry no new
     signal) and fit a straight line of logit(rate) against ln(1 + hours since
     launch), by weighted least squares with a 24-hour half-life, over the last
     48 hours of reads. The recent trend is the signal: the first day is dominated
     by dilution (players_tracked exploding pushes every rate down), the later
     reads by players actually progressing.
  2. Guard against Exophase's page-wide refresh jumps: if the median trophy's
     rate jumps by more than 1 logit between two consecutive distinct reads (the
     launch-night refresh moved most trophies ~10x down at once), drop every
     read before that jump. A level shift is not a trend, and fitting across one
     wrecks the slope.
  3. Shrink each trophy's slope toward the pooled (median) slope across trophies,
     so a noisy per-trophy fit cannot run away on its own.
  4. Spread: the house random-walk term (0.5 * sqrt(horizon in days)) combined
     with the fit's residual noise, with a floor, so the 90% band stays honest
     over a multi-day horizon.

An earners/players decomposition (fit log earners and log players tracked
separately, rate = earners / players) was tried and dropped: on the synthetic
backtest it consistently scored worse than the direct fit (dev/variants.py).

Standard library only, deterministic: no randomness, no wall clock, no network.
Reads every <id>.in in the input directory and writes one <id>.out per case.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper module
# shipped beside this file would import. (This entry is one file; nothing imports.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's term).
RW_SIGMA = 0.5
# Recent window and recency half-life for the trend fit, in hours.
WINDOW_HOURS = 48.0
HALF_LIFE_HOURS = 24.0
# Shrinkage strength: pooled-slope prior weight, in units of distinct points.
SHRINK_K = 5.0
# A page-wide jump bigger than this (median |logit change| across trophies)
# marks an Exophase refresh level shift; reads before it are dropped.
JUMP_LOGIT = 1.0
# Spread floor, in logit units: Exophase refreshes jump, so even a flat history
# does not mean a narrow future.
SPREAD_FLOOR = 0.5


def clamp_rate(pct):
    """Clamp a percent into the range the logit is defined on."""
    if pct is None:
        return None
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return min(99.99, max(0.01, v))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y):
    if y >= 20.0:
        return 99.99
    if y <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline's starter does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into epoch seconds."""
    if not text:
        return None
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def launch_epoch(case):
    """Midnight UTC of the game's release date, the log-time origin."""
    game = case.get("game") or {}
    day = game.get("release_date")
    if day:
        try:
            dt = datetime.strptime(str(day)[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    # Fall back to the first read's day at midnight.
    history = case.get("history") or []
    if history:
        first = parse_time(history[0].get("at"))
        if first is not None:
            return first - (first % 86400.0)
    return 0.0


def distinct_series(times, values):
    """Collapse consecutive duplicates: (time, value) at each change, plus the last."""
    out = []
    prev = None
    for t, v in zip(times, values):
        if v is None:
            continue
        if prev is None or v != prev:
            out.append((t, v))
            prev = v
    if out and out[-1][0] != times[-1] and values[-1] is not None:
        out.append((times[-1], values[-1]))
    return out


def median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b x. Returns (a, b, resid_sigma, n)."""
    n = len(xs)
    if n == 0:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if n < 3 or sxx <= 1e-12:
        return (my, 0.0, 0.0, n)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        sigma = math.sqrt(max(resid, 0.0))
    else:
        sigma = 0.0
    return (a, b, sigma, n)


def refresh_cutoff(times, rate_rows):
    """Time of the last page-wide Exophase refresh jump, or None.

    rate_rows: one filled rate vector per read, in trophy order. A jump is a step
    between two consecutive distinct page snapshots where the median trophy's
    |logit change| exceeds JUMP_LOGIT. Returns the time of the snapshot after
    the last such jump, so reads before it can be dropped.
    """
    snaps = []
    prev = None
    for t, row in zip(times, rate_rows):
        if row != prev:
            snaps.append((t, row))
            prev = row
    cutoff = None
    for (t0, r0), (t1, r1) in zip(snaps, snaps[1:]):
        jumps = []
        for a, b in zip(r0, r1):
            if a is None or b is None:
                continue
            jumps.append(abs(logit(b) - logit(a)))
        if jumps and median(jumps) > JUMP_LOGIT:
            cutoff = t1
    return cutoff


def fit_trend(points, t_last, x_target, t_min):
    """Fit logit(rate) on ln(1 + hours since launch) over the recent window.

    points: [(t_hours, rate)] distinct values. Only points at or after t_min
    (the later of the window start and any refresh-jump cutoff) are used.
    Returns (median_logit_at_target, slope, resid_sigma, n_points), or None.
    """
    if not points:
        return None
    window = [(t, r) for (t, r) in points if t >= t_min]
    if len(window) < 1:
        window = [points[-1]]
    xs = [math.log1p(t) for (t, _r) in window]
    ys = [logit(r) for (_t, r) in window]
    ws = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for (t, _r) in window]
    fit = wls_line(xs, ys, ws)
    if fit is None:
        return None
    a, b, sigma, n = fit
    return (a + b * x_target, b, sigma, n)


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError("%s asks for levels %s, not %s" % (name, case.get("levels"), LEVELS))
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    if not trophies:
        raise ValueError("%s has no trophies" % name)
    if not history:
        raise ValueError("%s has no history" % name)

    t0 = launch_epoch(case)
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target_at = parse_time(case.get("target"))
    if target_at is None:
        target_at = parse_time(case.get("at"))
    if target_at is None:
        target_at = parse_time(history[-1].get("at"))
    t_target = max(0.0, (target_at - t0) / 3600.0)
    x_target = math.log1p(t_target)

    # Per-read times (hours since launch) and filled rate vectors.
    read_times = []
    rate_rows = []
    for read in history:
        at = parse_time(read.get("at"))
        if at is None:
            at = parse_time(read.get("slot"))
        if at is None:
            continue
        read_times.append(max(0.0, (at - t0) / 3600.0))
        rs = read.get("rates") or []
        row = []
        last = None
        for i in range(len(trophies)):
            r = rs[i] if i < len(rs) else None
            if r is not None:
                last = r
            row.append(last)
        rate_rows.append(row)
    if not read_times:
        raise ValueError("%s has no dated reads" % name)
    t_last = read_times[-1]
    span_days = max((t_last - read_times[0]) / 24.0, 0.25)

    # Drop reads before the last page-wide refresh jump, if any.
    t_cut = refresh_cutoff(read_times, rate_rows)
    t_min = t_last - WINDOW_HOURS
    if t_cut is not None:
        t_min = max(t_min, t_cut)

    # First pass: direct per-trophy fits, to get the pooled slope.
    direct = []
    for i in range(len(trophies)):
        points = distinct_series(read_times, [row[i] for row in rate_rows])
        fit = fit_trend(points, t_last, x_target, t_min)
        direct.append((points, fit))

    slopes = [f[1] for (_p, f) in direct if f is not None and f[3] >= 3]
    pooled_slope = median(slopes) if slopes else 0.0

    spread_rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        points, fit = direct[i]
        if fit is None:
            last_rate = points[-1][1] if points else 0.01
            med_logit = logit(last_rate)
            sigma = max(spread_rw, SPREAD_FLOOR)
            rows.append((trophy["id"], med_logit, sigma))
            continue
        _med, slope, resid_sigma, n_pts = fit
        # Shrink the slope toward the pooled slope, re-anchoring the median to
        # the window's weighted mean so it stays tied to recent reads.
        n_eff = min(n_pts, 10)
        slope_s = (n_eff * slope + SHRINK_K * pooled_slope) / (n_eff + SHRINK_K)
        window = [(t, r) for (t, r) in points if t >= t_min]
        if len(window) < 1:
            window = [points[-1]]
        xs = [math.log1p(t) for (t, _r) in window]
        ys = [logit(r) for (_t, r) in window]
        ws = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for (t, _r) in window]
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        med_logit = my + slope_s * (x_target - mx)

        sigma_fit = min(resid_sigma, 0.5) * math.sqrt(
            max(horizon_days, 0.0) / span_days
        )
        sigma = math.sqrt(spread_rw ** 2 + sigma_fit ** 2 + 0.15 ** 2)
        sigma = max(sigma, SPREAD_FLOOR)
        rows.append((trophy["id"], med_logit, sigma))

    out = {"trophies": []}
    for tid, med_logit, sigma in rows:
        qs = []
        for z in Z:
            qs.append(round4(inv_logit(med_logit + z * sigma)))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out["trophies"].append({"id": tid, "q": qs})
    return out


def persistence_forecast(case, name):
    """Fallback the starter would write: last rate, random-walk spread."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    last = history[-1] if history else {"rates": []}
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    spread = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        rs = last.get("rates") or []
        rate = rs[i] if i < len(rs) and rs[i] is not None else 0.01
        lg = logit(rate)
        qs = [round4(inv_logit(lg + z * spread)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case invalid: fall back to persistence.
            forecast = persistence_forecast(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
