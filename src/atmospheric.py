#!/usr/bin/env python3
"""
BlackRoad Atmospheric Analyzer
Real atmospheric science: Magnus dew point, Rothfusz heat index,
barometric altitude, potential temperature, mixing ratio, UV index,
lifted index, weather classification and pressure trend analysis.
"""

import math
import sqlite3
import argparse
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ── Physical Constants ─────────────────────────────────────────────────────────
R_DRY    = 287.058    # J/(kg·K)  specific gas constant, dry air
R_VAPOR  = 461.495    # J/(kg·K)  specific gas constant, water vapour
CP_DRY   = 1005.7     # J/(kg·K)  specific heat at constant pressure, dry air
KAPPA    = R_DRY / CP_DRY          # ≈ 0.2854  Poisson constant
P0_PA    = 101325.0   # Pa        ISA sea-level pressure
P0_HPA   = 1013.25    # hPa       ISA sea-level pressure
T0_K     = 288.15     # K         ISA sea-level temperature (15 °C)
L_RATE   = 0.0065     # K/m       standard lapse rate
M_AIR    = 0.02896    # kg/mol    molar mass of dry air
G        = 9.80665    # m/s²      standard gravity
R_GAS    = 8.314462   # J/(mol·K) universal gas constant

# Magnus formula coefficients — Alduchov & Eskridge (1996)
MAGNUS_A = 17.625
MAGNUS_B = 243.04   # °C

# ANSI colour codes
RED     = "\033[91m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atmospheric.db")


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AtmosphericReading:
    """Single-station atmospheric observation."""
    temperature_c: float          # dry-bulb temperature °C
    relative_humidity: float      # 0–100 %
    pressure_hpa: float           # station pressure hPa
    altitude_m: float = 0.0       # metres above sea level
    wind_speed_ms: float = 0.0    # m/s
    solar_zenith_deg: float = 45.0# degrees from zenith
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class DewPointResult:
    dew_point_c: float
    depression_c: float     # T - Td
    frost_point_c: float
    formula: str = "Magnus (Alduchov & Eskridge 1996)"


@dataclass
class HeatIndexResult:
    heat_index_c: float
    apparent_temp_c: float
    danger_level: str
    formula: str = "Rothfusz regression (NWS SR 90-23)"


@dataclass
class UVIndexResult:
    uv_index: float
    exposure_category: str
    max_safe_minutes: int


@dataclass
class WeatherPattern:
    classification: str
    stability_index: float
    convective_available: bool
    description: str
    risk_level: str


@dataclass
class PressureTrend:
    readings: List[float]
    trend: str
    tendency_hpa_3h: float
    forecast: str


# ── Core Atmospheric Physics ───────────────────────────────────────────────────

def dew_point_magnus(T: float, RH: float) -> DewPointResult:
    """Magnus formula dew point.
    Td = B·γ / (A − γ)  where  γ = ln(RH/100) + A·T/(B+T)
    T in °C, RH in %.  Reference: Alduchov & Eskridge (1996).
    """
    if not (0 < RH <= 100):
        raise ValueError(f"RH must be in (0, 100], got {RH}")
    gamma = math.log(RH / 100.0) + (MAGNUS_A * T) / (MAGNUS_B + T)
    Td = (MAGNUS_B * gamma) / (MAGNUS_A - gamma)
    # Frost point: apply WMO-No.8 correction below 0 °C
    Tf = Td + 0.00422 * Td ** 2 if Td < 0 else Td
    return DewPointResult(
        dew_point_c=round(Td, 4),
        depression_c=round(T - Td, 4),
        frost_point_c=round(Tf, 4),
    )


def heat_index_rothfusz(T: float, RH: float) -> HeatIndexResult:
    """Rothfusz 9-coefficient polynomial heat index (NWS SR 90-23).
    T in °C.  The regression is applied in °F then converted back.
    NWS low-RH and high-RH corrections are included.
    """
    T_f = T * 9.0 / 5.0 + 32.0   # convert to °F
    c1 = -42.379
    c2 =   2.04901523
    c3 =  10.14333127
    c4 =  -0.22475541
    c5 =  -0.00683783
    c6 =  -0.05481717
    c7 =   0.00122874
    c8 =   0.00085282
    c9 =  -0.00000199

    HI_f = (c1
            + c2 * T_f
            + c3 * RH
            + c4 * T_f * RH
            + c5 * T_f ** 2
            + c6 * RH ** 2
            + c7 * T_f ** 2 * RH
            + c8 * T_f * RH ** 2
            + c9 * T_f ** 2 * RH ** 2)

    # NWS Adjustment 1: low humidity (RH < 13 %, 80 ≤ T ≤ 112 °F)
    if RH < 13 and 80.0 <= T_f <= 112.0:
        adj = ((13.0 - RH) / 4.0) * math.sqrt((17.0 - abs(T_f - 95.0)) / 17.0)
        HI_f -= adj
    # NWS Adjustment 2: high humidity (RH > 85 %, 80 ≤ T ≤ 87 °F)
    elif RH > 85.0 and 80.0 <= T_f <= 87.0:
        adj = ((RH - 85.0) / 10.0) * ((87.0 - T_f) / 5.0)
        HI_f += adj

    HI_c = (HI_f - 32.0) * 5.0 / 9.0

    # NWS danger categories
    if HI_c < 27.0:
        danger = "Comfortable"
    elif HI_c < 32.0:
        danger = "Caution"
    elif HI_c < 39.0:
        danger = "Extreme Caution"
    elif HI_c < 51.0:
        danger = "Danger"
    else:
        danger = "Extreme Danger"

    return HeatIndexResult(
        heat_index_c=round(HI_c, 2),
        apparent_temp_c=round(HI_c, 2),
        danger_level=danger,
    )


def wet_bulb_temperature(T: float, RH: float, P_hpa: float = 1013.25) -> float:
    """Psychrometric wet-bulb temperature via Stull (2011) polynomial.
    T °C, RH %.  Valid range: 5–99 % RH, −20 to 50 °C.
    Reference: Stull, R. (2011). J. Appl. Meteor. Climatol., 50, 2267-2269.
    """
    Tw = (T * math.atan(0.151977 * (RH + 8.313659) ** 0.5)
          + math.atan(T + RH)
          - math.atan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * math.atan(0.023101 * RH)
          - 4.686035)
    return round(Tw, 3)


def absolute_humidity(T: float, RH: float) -> float:
    """Absolute humidity ρ_v in g/m³.
    Saturation vapour pressure via August-Roche-Magnus; ideal-gas law for density.
    AH = e_actual × M_water / (R_gas × T_K) × 1000
    """
    e_sat = 6.1078 * math.exp((MAGNUS_A * T) / (MAGNUS_B + T))   # hPa
    e_act = (RH / 100.0) * e_sat                                   # hPa → Pa below
    M_water = 0.018016   # kg/mol
    T_K = T + 273.15
    AH = (e_act * 100.0 * M_water) / (R_GAS * T_K) * 1000.0       # g/m³
    return round(AH, 4)


def barometric_pressure_altitude(P_hpa: float,
                                  T_k: float = T0_K,
                                  P0_hpa: float = P0_HPA) -> float:
    """Altitude from the hypsometric (barometric) equation.
    P = P0 · exp(−M·g·h / (R·T))  ⟹  h = −(R·T)/(M·g) · ln(P/P0)
    Returns height in metres.
    """
    if P_hpa <= 0.0 or P0_hpa <= 0.0:
        raise ValueError("Pressure must be strictly positive")
    h = -(R_GAS * T_k / (M_AIR * G)) * math.log(P_hpa / P0_hpa)
    return round(h, 1)


def potential_temperature(T_c: float, P_hpa: float) -> float:
    """Potential temperature θ = T · (P₀/P)^(R/cₚ).
    T in °C, P in hPa.  Returns θ in °C.
    κ = R_dry / cp_dry ≈ 0.2854 (Poisson constant).
    """
    T_k = T_c + 273.15
    theta_k = T_k * (P0_HPA / P_hpa) ** KAPPA
    return round(theta_k - 273.15, 4)


def mixing_ratio(T_c: float, RH: float, P_hpa: float) -> float:
    """Water-vapour mixing ratio w = ε · e / (P − e)  [kg/kg].
    ε = R_dry / R_vapour = 0.6220.
    e_sat from Magnus formula.
    """
    epsilon = R_DRY / R_VAPOR   # 0.6220
    e_sat = 6.1078 * math.exp((MAGNUS_A * T_c) / (MAGNUS_B + T_c))  # hPa
    e = (RH / 100.0) * e_sat
    w = epsilon * e / (P_hpa - e)
    return round(w, 6)


def specific_humidity(T_c: float, RH: float, P_hpa: float) -> float:
    """Specific humidity q = w / (1 + w)  [kg/kg]."""
    w = mixing_ratio(T_c, RH, P_hpa)
    return round(w / (1.0 + w), 6)


def uv_index(solar_zenith_deg: float,
             ozone_du: float = 300.0,
             cloud_factor: float = 1.0) -> UVIndexResult:
    """UV index from solar zenith angle, total ozone, and cloud attenuation.
    Proportional to cos(zenith) / (ozone/300)^1.2, scaled to WHO UVI scale.
    ozone_du: total column ozone in Dobson Units (typical: 250–400 DU).
    cloud_factor: 1.0 = clear, 0.3 = overcast.
    """
    if solar_zenith_deg >= 90.0:
        uvi = 0.0
    else:
        cos_z = math.cos(math.radians(solar_zenith_deg))
        ozone_factor = (300.0 / ozone_du) ** 1.2
        uvi = 40.0 * cos_z * ozone_factor * cloud_factor
        uvi = max(0.0, uvi)
    uvi = round(uvi, 1)

    if uvi < 3.0:
        cat, safe_min = "Low",       60
    elif uvi < 6.0:
        cat, safe_min = "Moderate",  30
    elif uvi < 8.0:
        cat, safe_min = "High",      20
    elif uvi < 11.0:
        cat, safe_min = "Very High", 10
    else:
        cat, safe_min = "Extreme",    5

    return UVIndexResult(uv_index=uvi,
                         exposure_category=cat,
                         max_safe_minutes=safe_min)


def lifted_index(T_surface_c: float,
                 Td_surface_c: float,
                 T_500hpa_c: float) -> float:
    """Lifted Index LI = T_environment_500hPa − T_parcel_500hPa.
    Negative LI → instability; LI < −6 → severe convective risk.
    Parcel ascends dry-adiabatically to LCL, then moist-adiabatically to 500 hPa.
    LCL height rule: h_lcl ≈ 125 · (T − Td) metres.
    """
    h_lcl = 125.0 * (T_surface_c - Td_surface_c)          # metres
    h_500 = barometric_pressure_altitude(500.0)             # ≈ 5574 m
    # LCL temperature: parcel cools at DALR (9.8 K/km) to LCL
    T_lcl = T_surface_c - 0.0098 * h_lcl
    # Above LCL parcel follows MALR ≈ 6.5 K/km (simplified constant)
    MALR = 0.0065
    h_above_lcl = max(0.0, h_500 - h_lcl)
    T_parcel_500 = T_lcl - MALR * h_above_lcl
    LI = T_500hpa_c - T_parcel_500
    return round(LI, 2)


# ── Weather Classification ─────────────────────────────────────────────────────

def classify_weather(reading: AtmosphericReading) -> WeatherPattern:
    """Multi-parameter synoptic weather classification.
    Uses dew point, lifted index, potential temperature, and pressure anomaly.
    """
    dp = dew_point_magnus(reading.temperature_c, reading.relative_humidity)
    # Crude 500 hPa temperature estimate: −25 K from surface
    T_500_estimate = reading.temperature_c - 25.0
    li = lifted_index(reading.temperature_c, dp.dew_point_c, T_500_estimate)
    p_anomaly = reading.pressure_hpa - P0_HPA

    # Composite stability index (positive = more stable)
    stability_index = li + p_anomaly / 10.0

    if reading.relative_humidity > 85.0 and reading.pressure_hpa < 1000.0:
        cls  = "Frontal / Cyclonic"
        risk = "High"
        desc = "Deep low pressure with high humidity — frontal passage likely."
        conv = True
    elif li < -6.0:
        cls  = "Severe Convective"
        risk = "High"
        desc = "Highly unstable. Severe thunderstorms and large hail possible."
        conv = True
    elif li < -3.0:
        cls  = "Convective / Thunderstorm"
        risk = "High"
        desc = "Strong convective instability. Thunderstorms probable."
        conv = True
    elif li < 0.0 and reading.relative_humidity > 60.0:
        cls  = "Unstable"
        risk = "Moderate"
        desc = "Conditionally unstable. Shower and storm development possible."
        conv = True
    elif reading.relative_humidity < 40.0 and reading.pressure_hpa > 1020.0:
        cls  = "Stable / High Pressure"
        risk = "Low"
        desc = "Anticyclonic conditions. Clear and settled weather."
        conv = False
    elif abs(p_anomaly) < 5.0 and li >= 0.0:
        cls  = "Neutral"
        risk = "Low"
        desc = "Near-average pressure. Variable but non-severe conditions."
        conv = False
    else:
        cls  = "Stable"
        risk = "Low"
        desc = "Stable atmosphere. No significant convection expected."
        conv = False

    return WeatherPattern(
        classification=cls,
        stability_index=round(stability_index, 2),
        convective_available=conv,
        description=desc,
        risk_level=risk,
    )


def pressure_trend(readings: List[float]) -> PressureTrend:
    """3-hour pressure tendency classification.
    Tendency = total change normalised to 3-hour equivalent.
    WMO SYNOP tendency codes:
      +1.5 hPa/3h  rising rapidly
       +0.5        rising
       −0.5        falling
       −1.5        falling rapidly
    """
    if len(readings) < 2:
        return PressureTrend(
            readings=readings, trend="insufficient data",
            tendency_hpa_3h=0.0, forecast="Unknown — need more data"
        )
    delta = readings[-1] - readings[0]
    tendency = delta * 3.0 / len(readings)   # normalise to 3-h equivalent

    if tendency > 1.5:
        trend, forecast = "Rapidly Rising",  "Improving rapidly — high pressure building"
    elif tendency > 0.5:
        trend, forecast = "Rising",          "Fair weather likely"
    elif tendency < -1.5:
        trend, forecast = "Rapidly Falling", "Rapid deterioration — storm possible"
    elif tendency < -0.5:
        trend, forecast = "Falling",         "Cloudy with rain likely"
    else:
        trend, forecast = "Steady",          "Little change expected"

    return PressureTrend(
        readings=readings,
        trend=trend,
        tendency_hpa_3h=round(tendency, 2),
        forecast=forecast,
    )


# ── SQLite Persistence ─────────────────────────────────────────────────────────

def _init_db(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            temperature_c     REAL    NOT NULL,
            relative_humidity REAL    NOT NULL,
            pressure_hpa      REAL    NOT NULL,
            altitude_m        REAL    DEFAULT 0,
            wind_speed_ms     REAL    DEFAULT 0,
            solar_zenith      REAL    DEFAULT 45,
            dew_point_c       REAL,
            heat_index_c      REAL,
            wet_bulb_c        REAL,
            abs_humidity      REAL,
            potential_temp    REAL,
            mixing_ratio      REAL,
            uv_index          REAL,
            classification    TEXT
        )
    """)
    conn.commit()
    return conn


def save_reading(reading: AtmosphericReading, db_path: str = DB_PATH) -> int:
    """Compute all derived parameters and persist the reading."""
    dp  = dew_point_magnus(reading.temperature_c, reading.relative_humidity)
    hi  = heat_index_rothfusz(reading.temperature_c, reading.relative_humidity)
    wb  = wet_bulb_temperature(reading.temperature_c, reading.relative_humidity,
                                reading.pressure_hpa)
    ah  = absolute_humidity(reading.temperature_c, reading.relative_humidity)
    pt  = potential_temperature(reading.temperature_c, reading.pressure_hpa)
    mr  = mixing_ratio(reading.temperature_c, reading.relative_humidity,
                       reading.pressure_hpa)
    uv  = uv_index(reading.solar_zenith_deg)
    pat = classify_weather(reading)

    conn = _init_db(db_path)
    cur  = conn.execute("""
        INSERT INTO readings
            (timestamp, temperature_c, relative_humidity, pressure_hpa,
             altitude_m, wind_speed_ms, solar_zenith,
             dew_point_c, heat_index_c, wet_bulb_c, abs_humidity,
             potential_temp, mixing_ratio, uv_index, classification)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (reading.timestamp, reading.temperature_c, reading.relative_humidity,
          reading.pressure_hpa, reading.altitude_m, reading.wind_speed_ms,
          reading.solar_zenith_deg, dp.dew_point_c, hi.heat_index_c, wb,
          ah, pt, mr, uv.uv_index, pat.classification))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def list_readings(limit: int = 20, db_path: str = DB_PATH) -> List[dict]:
    conn = _init_db(db_path)
    cur  = conn.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def get_pressure_series(limit: int = 24, db_path: str = DB_PATH) -> List[float]:
    conn = _init_db(db_path)
    cur  = conn.execute(
        "SELECT pressure_hpa FROM readings ORDER BY id ASC LIMIT ?", (limit,))
    vals = [r[0] for r in cur.fetchall()]
    conn.close()
    return vals


# ── ASCII Visualisation ────────────────────────────────────────────────────────

def ascii_trend_chart(values: List[float], label: str = "Value",
                      width: int = 50, height: int = 10) -> str:
    """Render a text-mode line chart with ANSI colours."""
    if not values:
        return "(no data)"
    lo   = min(values)
    hi_v = max(values)
    span = hi_v - lo or 1.0
    n    = min(len(values), width)
    vals = values[-n:]

    lines = [
        f"\n{CYAN}{BOLD}{label} Trend Chart{RESET}",
        f"  {YELLOW}Max: {hi_v:.2f}  Min: {lo:.2f}  Δ: {span:.2f}{RESET}",
        "",
    ]

    grid = [[" "] * n for _ in range(height)]
    for x, v in enumerate(vals):
        y = int((v - lo) / span * (height - 1))
        y = height - 1 - y
        grid[y][x] = "●"
    # Connect vertically between adjacent points
    for x in range(1, n):
        y_prev = height - 1 - int((vals[x - 1] - lo) / span * (height - 1))
        y_cur  = height - 1 - int((vals[x]     - lo) / span * (height - 1))
        lo_y, hi_y = sorted((y_prev, y_cur))
        for y in range(lo_y + 1, hi_y):
            if grid[y][x] == " ":
                grid[y][x] = "│"

    for row_i, row in enumerate(grid):
        row_val = hi_v - row_i / (height - 1) * span if height > 1 else hi_v
        prefix  = f"{row_val:8.2f} │"
        lines.append(prefix + "".join(row))

    lines.append("         └" + "─" * n)
    return "\n".join(lines)


def colour_temperature(T: float) -> str:
    """Return ANSI-coloured temperature string."""
    if T < 0.0:
        return f"{BLUE}{T:.1f}°C{RESET}"
    if T < 10.0:
        return f"{CYAN}{T:.1f}°C{RESET}"
    if T < 25.0:
        return f"{GREEN}{T:.1f}°C{RESET}"
    if T < 35.0:
        return f"{YELLOW}{T:.1f}°C{RESET}"
    return f"{RED}{T:.1f}°C{RESET}"


# ── Full Analysis Report ───────────────────────────────────────────────────────

def full_report(reading: AtmosphericReading) -> str:
    """Generate a comprehensive formatted atmospheric analysis report."""
    dp  = dew_point_magnus(reading.temperature_c, reading.relative_humidity)
    hi  = heat_index_rothfusz(reading.temperature_c, reading.relative_humidity)
    wb  = wet_bulb_temperature(reading.temperature_c, reading.relative_humidity,
                                reading.pressure_hpa)
    ah  = absolute_humidity(reading.temperature_c, reading.relative_humidity)
    pt  = potential_temperature(reading.temperature_c, reading.pressure_hpa)
    mr  = mixing_ratio(reading.temperature_c, reading.relative_humidity,
                       reading.pressure_hpa)
    sh  = specific_humidity(reading.temperature_c, reading.relative_humidity,
                            reading.pressure_hpa)
    uv  = uv_index(reading.solar_zenith_deg)
    alt = barometric_pressure_altitude(reading.pressure_hpa)
    pat = classify_weather(reading)
    li  = lifted_index(reading.temperature_c, dp.dew_point_c,
                       reading.temperature_c - 25.0)

    lines = [
        f"\n{BOLD}{CYAN}{'═' * 62}{RESET}",
        f"{BOLD}{CYAN}   BlackRoad Atmospheric Analyzer — Full Report{RESET}",
        f"{BOLD}{CYAN}{'═' * 62}{RESET}",
        f"  Timestamp        : {reading.timestamp}",
        f"  Temperature      : {colour_temperature(reading.temperature_c)}",
        f"  Relative Humidity: {YELLOW}{reading.relative_humidity:.1f}%{RESET}",
        f"  Station Pressure : {MAGENTA}{reading.pressure_hpa:.1f} hPa{RESET}",
        f"  Derived Altitude : {alt:.0f} m  (hypsometric)",
        f"  Wind Speed       : {reading.wind_speed_ms:.1f} m/s",
        f"  Solar Zenith     : {reading.solar_zenith_deg:.1f}°",
        f"",
        f"{BOLD}── Thermodynamic Parameters ──────────────────────────────{RESET}",
        f"  Dew Point        : {colour_temperature(dp.dew_point_c)}"
        f"  (depression {dp.depression_c:.1f} °C)",
        f"  Frost Point      : {dp.frost_point_c:.2f}°C",
        f"  Wet-Bulb Temp    : {wb:.2f}°C",
        f"  Heat Index       : {colour_temperature(hi.heat_index_c)}"
        f"  [{hi.danger_level}]",
        f"  Potential Temp θ : {pt:.2f}°C",
        f"",
        f"{BOLD}── Moisture Parameters ───────────────────────────────────{RESET}",
        f"  Absolute Humidity: {ah:.2f} g/m³",
        f"  Mixing Ratio     : {mr * 1000:.3f} g/kg",
        f"  Specific Humidity: {sh * 1000:.3f} g/kg",
        f"",
        f"{BOLD}── Stability & Convection ────────────────────────────────{RESET}",
        f"  Lifted Index     : {li:+.2f}"
        f"  ({'unstable' if li < 0 else 'stable'})",
        f"  Pattern          : {BOLD}{pat.classification}{RESET}",
        f"  Risk Level       : "
        f"{RED if pat.risk_level == 'High' else YELLOW if pat.risk_level == 'Moderate' else GREEN}"
        f"{pat.risk_level}{RESET}",
        f"  Description      : {pat.description}",
        f"",
        f"{BOLD}── UV Radiation ──────────────────────────────────────────{RESET}",
        f"  UV Index         : {uv.uv_index}  [{YELLOW}{uv.exposure_category}{RESET}]",
        f"  Max safe exposure: {uv.max_safe_minutes} minutes",
        f"{CYAN}{'═' * 62}{RESET}\n",
    ]
    return "\n".join(lines)


# ── CLI Commands ───────────────────────────────────────────────────────────────

def _add_met_args(p: argparse.ArgumentParser) -> None:
    """Add standard meteorological input arguments to a subparser."""
    p.add_argument("-T", "--temperature", type=float, required=True,
                   metavar="°C",  help="Dry-bulb temperature in °C")
    p.add_argument("-r", "--humidity",    type=float, required=True,
                   metavar="%",   help="Relative humidity 0–100 %%")
    p.add_argument("-p", "--pressure",    type=float, default=1013.25,
                   metavar="hPa", help="Station pressure hPa (default 1013.25)")
    p.add_argument("-a", "--altitude",    type=float, default=0.0,
                   metavar="m",   help="Station altitude m (default 0)")
    p.add_argument("-w", "--wind",        type=float, default=0.0,
                   metavar="m/s", help="Wind speed m/s (default 0)")
    p.add_argument("-z", "--zenith",      type=float, default=45.0,
                   metavar="deg", help="Solar zenith angle degrees (default 45)")


def cmd_analyze(args: argparse.Namespace) -> None:
    r = AtmosphericReading(
        temperature_c=args.temperature, relative_humidity=args.humidity,
        pressure_hpa=args.pressure, altitude_m=args.altitude,
        wind_speed_ms=args.wind, solar_zenith_deg=args.zenith,
    )
    print(full_report(r))


def cmd_record(args: argparse.Namespace) -> None:
    r = AtmosphericReading(
        temperature_c=args.temperature, relative_humidity=args.humidity,
        pressure_hpa=args.pressure, altitude_m=args.altitude,
        wind_speed_ms=args.wind, solar_zenith_deg=args.zenith,
    )
    row_id = save_reading(r)
    print(f"{GREEN}✓ Saved as record ID {row_id}{RESET}")
    print(full_report(r))


def cmd_list(args: argparse.Namespace) -> None:
    rows = list_readings(limit=args.limit)
    if not rows:
        print(f"{YELLOW}No readings stored yet. Use 'record' to add data.{RESET}")
        return
    hdr = (f"{'ID':>4}  {'Timestamp':>23}  {'T(°C)':>7}  "
           f"{'RH%':>5}  {'P(hPa)':>7}  Pattern")
    print(f"\n{BOLD}{hdr}{RESET}")
    print("─" * 80)
    for row in rows:
        print(f"{row['id']:>4}  {row['timestamp']:>23}  "
              f"{row['temperature_c']:>7.1f}  {row['relative_humidity']:>5.1f}  "
              f"{row['pressure_hpa']:>7.1f}  {row['classification']}")


def cmd_trends(args: argparse.Namespace) -> None:
    series = get_pressure_series(limit=args.limit)
    if len(series) < 2:
        print(f"{YELLOW}Need ≥ 2 readings for trend analysis.{RESET}")
        return
    pt = pressure_trend(series)
    print(f"\n{BOLD}Pressure Trend  : {CYAN}{pt.trend}{RESET}")
    print(f"3-h tendency    : {pt.tendency_hpa_3h:+.2f} hPa")
    print(f"Forecast        : {pt.forecast}")
    print(ascii_trend_chart(series, label="Pressure (hPa)"))


def cmd_classify(args: argparse.Namespace) -> None:
    r = AtmosphericReading(
        temperature_c=args.temperature, relative_humidity=args.humidity,
        pressure_hpa=args.pressure,
    )
    pat = classify_weather(r)
    dp  = dew_point_magnus(args.temperature, args.humidity)
    li  = lifted_index(args.temperature, dp.dew_point_c, args.temperature - 25.0)
    print(f"\n{BOLD}Weather Classification{RESET}")
    print(f"  Pattern    : {BOLD}{pat.classification}{RESET}")
    print(f"  Risk Level : {pat.risk_level}")
    print(f"  Stability  : {pat.stability_index:.2f}")
    print(f"  Lifted Idx : {li:+.2f}")
    print(f"  Convective : {'Yes' if pat.convective_available else 'No'}")
    print(f"  Details    : {pat.description}")


def cmd_report(args: argparse.Namespace) -> None:
    """Statistical summary across stored readings."""
    rows = list_readings(limit=args.limit)
    if not rows:
        print(f"{YELLOW}No data available. Use 'record' first.{RESET}")
        return
    temps     = [r["temperature_c"]     for r in rows]
    humids    = [r["relative_humidity"] for r in rows]
    pressures = [r["pressure_hpa"]      for r in rows]

    def stats(v: List[float]) -> str:
        if len(v) < 2:
            return f"mean={statistics.mean(v):.2f}"
        return (f"mean={statistics.mean(v):.2f}  "
                f"min={min(v):.2f}  max={max(v):.2f}  "
                f"σ={statistics.stdev(v):.2f}")

    print(f"\n{BOLD}{CYAN}Statistical Report — last {len(rows)} readings{RESET}")
    print(f"  Temperature  : {stats(temps)} °C")
    print(f"  Humidity     : {stats(humids)} %")
    print(f"  Pressure     : {stats(pressures)} hPa")

    pt = pressure_trend(pressures)
    print(f"  P-Trend      : {pt.trend}  ({pt.tendency_hpa_3h:+.2f} hPa/3h)")
    print(f"  Forecast     : {pt.forecast}")
    print(ascii_trend_chart(temps,     label="Temperature (°C)"))
    print(ascii_trend_chart(pressures, label="Pressure (hPa)"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atmospheric",
        description=(
            "BlackRoad Atmospheric Analyzer — "
            "real atmospheric science in the terminal"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  atmospheric analyze -T 28.5 -r 72 -p 1008.3 -z 35\n"
            "  atmospheric record  -T 22   -r 60 -p 1013\n"
            "  atmospheric list -n 10\n"
            "  atmospheric trends\n"
            "  atmospheric classify -T 30 -r 88 -p 995\n"
            "  atmospheric report -n 50\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze",  help="Full atmospheric analysis (no save)")
    _add_met_args(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_record = sub.add_parser("record", help="Analyse and persist to SQLite")
    _add_met_args(p_record)
    p_record.set_defaults(func=cmd_record)

    p_list = sub.add_parser("list", help="List stored readings")
    p_list.add_argument("-n", "--limit", type=int, default=20,
                        help="Maximum rows to show (default 20)")
    p_list.set_defaults(func=cmd_list)

    p_trends = sub.add_parser("trends", help="ASCII pressure trend chart")
    p_trends.add_argument("-n", "--limit", type=int, default=24,
                           help="Number of readings to include")
    p_trends.set_defaults(func=cmd_trends)

    p_cls = sub.add_parser("classify", help="Weather pattern classification")
    _add_met_args(p_cls)
    p_cls.set_defaults(func=cmd_classify)

    p_rep = sub.add_parser("report", help="Statistical summary of stored readings")
    p_rep.add_argument("-n", "--limit", type=int, default=50,
                       help="Number of readings (default 50)")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    build_parser().parse_args().__dict__.pop("func").__call__(
        build_parser().parse_args())


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
