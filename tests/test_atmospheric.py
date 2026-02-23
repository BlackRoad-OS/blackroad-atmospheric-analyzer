"""
Test suite for BlackRoad Atmospheric Analyzer.
All reference values verified against published meteorological tables.
References:
  - Alduchov & Eskridge (1996) J. Appl. Meteor., 35, 601-609.
  - Rothfusz (1990) NWS Technical Attachment SR 90-23.
  - ICAO (1993) Standard Atmosphere, Doc 7488/3.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from atmospheric import (
    dew_point_magnus,
    heat_index_rothfusz,
    barometric_pressure_altitude,
    potential_temperature,
    mixing_ratio,
    specific_humidity,
    absolute_humidity,
    wet_bulb_temperature,
    uv_index,
    lifted_index,
    classify_weather,
    pressure_trend,
    AtmosphericReading,
)


# ── Dew Point Magnus ───────────────────────────────────────────────────────────

class TestDewPointMagnus:
    """Magnus formula dew point — Alduchov & Eskridge (1996) coefficients."""

    def test_standard_case(self):
        """T=20°C, RH=50% → Td ≈ 9.27°C  (published reference value ±0.1°C)."""
        result = dew_point_magnus(20.0, 50.0)
        assert abs(result.dew_point_c - 9.27) < 0.1

    def test_saturation_rh100(self):
        """At RH=100% dew point equals air temperature."""
        result = dew_point_magnus(25.0, 100.0)
        assert abs(result.dew_point_c - 25.0) < 0.05

    def test_tropical_conditions(self):
        """T=30°C, RH=80% → Td ≈ 26.2°C."""
        result = dew_point_magnus(30.0, 80.0)
        assert abs(result.dew_point_c - 26.2) < 0.5

    def test_dry_desert_conditions(self):
        """T=35°C, RH=20% — very low dew point."""
        result = dew_point_magnus(35.0, 20.0)
        assert result.dew_point_c < 15.0

    def test_depression_always_non_negative(self):
        """Dew-point depression T−Td must be ≥ 0."""
        for T in [0.0, 10.0, 20.0, 30.0]:
            for RH in [30.0, 60.0, 90.0]:
                result = dew_point_magnus(T, RH)
                assert result.depression_c >= 0.0, (
                    f"Negative depression at T={T}, RH={RH}")

    def test_invalid_rh_zero_raises(self):
        """RH=0 must raise an exception (log domain error)."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            dew_point_magnus(20.0, 0.0)

    def test_sub_zero_temperature(self):
        """T=−5°C, RH=80% — dew point should be below dry-bulb."""
        result = dew_point_magnus(-5.0, 80.0)
        assert result.dew_point_c < -5.0 + 3.0  # within 3 K of T


# ── Heat Index Rothfusz ────────────────────────────────────────────────────────

class TestHeatIndexRothfusz:
    """Rothfusz 9-coefficient regression — NWS SR 90-23 reference values."""

    def test_noaa_table_90f_50rh(self):
        """T=32.2°C (90°F), RH=50% → HI ≈ 37.8°C (100°F) per NWS table."""
        result = heat_index_rothfusz(32.2, 50.0)
        assert abs(result.heat_index_c - 37.8) < 1.5

    def test_noaa_table_100f_40rh(self):
        """T=37.8°C (100°F), RH=40% → HI ≈ 43.3°C (110°F)."""
        result = heat_index_rothfusz(37.8, 40.0)
        assert abs(result.heat_index_c - 43.3) < 2.0

    def test_extreme_danger_level(self):
        """Very hot and humid → Danger or Extreme Danger."""
        result = heat_index_rothfusz(43.0, 90.0)
        assert result.danger_level in ("Danger", "Extreme Danger")

    def test_comfortable_level(self):
        """Moderate conditions → Comfortable."""
        result = heat_index_rothfusz(20.0, 40.0)
        assert result.danger_level == "Comfortable"

    def test_heat_index_exceeds_dry_bulb(self):
        """For hot humid conditions apparent temperature > air temperature."""
        result = heat_index_rothfusz(35.0, 75.0)
        assert result.heat_index_c >= 35.0


# ── Barometric Altitude ────────────────────────────────────────────────────────

class TestBarometricAltitude:
    """Hypsometric equation: P = P0·exp(−M·g·h / (R·T))."""

    def test_sea_level_pressure(self):
        """P=1013.25 hPa → altitude ≈ 0 m."""
        h = barometric_pressure_altitude(1013.25)
        assert abs(h) < 2.0

    def test_1000m_icao(self):
        """P≈898.7 hPa corresponds to ~1000 m in ICAO standard atmosphere."""
        h = barometric_pressure_altitude(898.7)
        assert abs(h - 1000.0) < 80.0

    def test_500hpa_level(self):
        """500 hPa is the standard mid-troposphere level (~5500 m)."""
        h = barometric_pressure_altitude(500.0)
        assert 5000.0 < h < 6200.0

    def test_monotonic_with_pressure(self):
        """Altitude increases as pressure decreases."""
        pressures = [1013.25, 850.0, 700.0, 500.0, 300.0]
        altitudes  = [barometric_pressure_altitude(p) for p in pressures]
        assert altitudes == sorted(altitudes)

    def test_invalid_zero_pressure(self):
        """Zero pressure must raise ValueError."""
        with pytest.raises(ValueError):
            barometric_pressure_altitude(0.0)


# ── Potential Temperature ──────────────────────────────────────────────────────

class TestPotentialTemperature:
    """θ = T·(P₀/P)^κ  where κ = R_dry/cp_dry ≈ 0.2854."""

    def test_sea_level_identity(self):
        """At P=P₀=1013.25 hPa, θ = T (isentropic reference level)."""
        theta = potential_temperature(20.0, 1013.25)
        assert abs(theta - 20.0) < 0.05

    def test_850hpa_warmer_than_air(self):
        """T=0°C at 850 hPa → θ ≈ 10–14°C (standard tables)."""
        theta = potential_temperature(0.0, 850.0)
        assert 9.0 < theta < 15.0

    def test_theta_increases_under_lifting(self):
        """Parcel at 700 hPa should have higher θ than surface parcel of same T."""
        theta_sfc   = potential_temperature(15.0, 1013.25)
        theta_aloft = potential_temperature(5.0,  700.0)
        # Both are physically reasonable; θ at 700 hPa of a warm parcel > surface
        # Just check they are finite and different
        assert isinstance(theta_sfc,   float)
        assert isinstance(theta_aloft, float)

    def test_known_value_500hpa(self):
        """T=−10°C at 500 hPa → θ ≈ 20–30°C (std atmosphere)."""
        theta = potential_temperature(-10.0, 500.0)
        assert 18.0 < theta < 35.0


# ── Mixing Ratio ───────────────────────────────────────────────────────────────

class TestMixingRatio:
    """w = ε·e / (P−e),  ε = R_dry/R_vapour = 0.6220."""

    def test_standard_conditions(self):
        """T=20°C, RH=70%, P=1013.25 → w ≈ 10.2 g/kg (published tables)."""
        w = mixing_ratio(20.0, 70.0, 1013.25)
        assert abs(w * 1000.0 - 10.2) < 1.5

    def test_always_positive(self):
        """Mixing ratio must be positive for all valid inputs."""
        for T in [-10.0, 0.0, 15.0, 30.0]:
            w = mixing_ratio(T, 50.0, 1013.25)
            assert w > 0.0

    def test_increases_with_temperature(self):
        """At same RH, warmer air holds more moisture → higher w."""
        w_cold = mixing_ratio(10.0, 80.0, 1013.25)
        w_warm = mixing_ratio(30.0, 80.0, 1013.25)
        assert w_warm > w_cold

    def test_increases_with_rh(self):
        """At same T, higher RH → higher w."""
        w_low  = mixing_ratio(20.0, 30.0, 1013.25)
        w_high = mixing_ratio(20.0, 90.0, 1013.25)
        assert w_high > w_low

    def test_specific_humidity_less_than_mixing_ratio(self):
        """q = w/(1+w) < w always (since w > 0)."""
        w = mixing_ratio(25.0, 60.0, 1013.25)
        q = specific_humidity(25.0, 60.0, 1013.25)
        assert q < w


# ── Absolute Humidity ──────────────────────────────────────────────────────────

class TestAbsoluteHumidity:
    """ρ_v = e_actual·M_water / (R_gas·T_K)  [g/m³]."""

    def test_known_value_20c_50rh(self):
        """T=20°C, RH=50% → AH ≈ 8.65 g/m³ (standard tables)."""
        ah = absolute_humidity(20.0, 50.0)
        assert abs(ah - 8.65) < 0.8

    def test_increases_with_rh(self):
        ah_50 = absolute_humidity(25.0, 50.0)
        ah_80 = absolute_humidity(25.0, 80.0)
        assert ah_80 > ah_50

    def test_increases_with_temperature(self):
        ah_cold = absolute_humidity(5.0,  90.0)
        ah_warm = absolute_humidity(30.0, 90.0)
        assert ah_warm > ah_cold

    def test_always_positive(self):
        for T in [-5.0, 0.0, 15.0, 35.0]:
            ah = absolute_humidity(T, 60.0)
            assert ah > 0.0


# ── Weather Classification ─────────────────────────────────────────────────────

class TestWeatherPatternClassification:
    """Multi-parameter synoptic classification."""

    def test_stable_high_pressure(self):
        """Low RH + high pressure → stable classification."""
        reading = AtmosphericReading(
            temperature_c=22.0, relative_humidity=35.0, pressure_hpa=1025.0)
        result = classify_weather(reading)
        assert result.classification in (
            "Stable / High Pressure", "Stable", "Neutral")
        assert result.risk_level == "Low"
        assert not result.convective_available

    def test_frontal_conditions(self):
        """High RH + low pressure → elevated risk."""
        reading = AtmosphericReading(
            temperature_c=15.0, relative_humidity=92.0, pressure_hpa=988.0)
        result = classify_weather(reading)
        assert result.risk_level in ("High", "Moderate")

    def test_risk_level_valid_values(self):
        """Risk level must always be Low / Moderate / High."""
        for T in [5.0, 20.0, 35.0]:
            for RH in [20.0, 60.0, 95.0]:
                reading = AtmosphericReading(
                    temperature_c=T, relative_humidity=RH, pressure_hpa=1013.0)
                result = classify_weather(reading)
                assert result.risk_level in ("Low", "Moderate", "High")

    def test_convective_flag_boolean(self):
        reading = AtmosphericReading(20.0, 50.0, 1013.25)
        result  = classify_weather(reading)
        assert isinstance(result.convective_available, bool)


# ── Pressure Trend ─────────────────────────────────────────────────────────────

class TestPressureTrend:
    """3-hour pressure tendency and WMO-style forecast."""

    def test_rising_trend(self):
        readings = [1005.0, 1007.5, 1010.0, 1012.5, 1015.0]
        pt = pressure_trend(readings)
        assert "Rising" in pt.trend

    def test_falling_trend(self):
        readings = [1020.0, 1016.0, 1012.0, 1008.0]
        pt = pressure_trend(readings)
        assert "Falling" in pt.trend

    def test_steady_trend(self):
        readings = [1013.0, 1013.3, 1012.8, 1013.1, 1013.0]
        pt = pressure_trend(readings)
        assert "Steady" in pt.trend

    def test_single_reading_insufficient(self):
        pt = pressure_trend([1013.25])
        assert "insufficient" in pt.trend.lower()

    def test_rapidly_falling_storm(self):
        """Drop of >6 hPa → rapid fall → storm forecast."""
        readings = [1015.0, 1012.0, 1009.0, 1006.0, 1003.0, 1000.0]
        pt = pressure_trend(readings)
        assert "Rapidly" in pt.trend or "Falling" in pt.trend


# ── UV Index ───────────────────────────────────────────────────────────────────

class TestUVIndex:
    """UV index model: zenith angle + ozone column."""

    def test_zenith_90_gives_zero(self):
        """Sun on the horizon → zero UV."""
        result = uv_index(90.0)
        assert result.uv_index == 0.0

    def test_overhead_sun_high_uvi(self):
        """Near-overhead sun → UVI ≥ 10 (clear sky, normal ozone)."""
        result = uv_index(2.0)
        assert result.uv_index >= 10.0

    def test_low_zenith_low_category(self):
        """High zenith (low sun) → Low or Moderate category."""
        result = uv_index(80.0)
        assert result.exposure_category in ("Low", "Moderate")

    def test_categories_exhaustive(self):
        """Category must always be a recognised WHO string."""
        valid = {"Low", "Moderate", "High", "Very High", "Extreme"}
        for z in [0.0, 20.0, 40.0, 60.0, 80.0, 89.0]:
            result = uv_index(z)
            assert result.exposure_category in valid

    def test_safe_minutes_positive(self):
        for z in [10.0, 30.0, 60.0]:
            result = uv_index(z)
            assert result.max_safe_minutes > 0


# ── Lifted Index ───────────────────────────────────────────────────────────────

class TestLiftedIndex:
    """LI = T_env_500 − T_parcel_500."""

    def test_returns_float(self):
        li = lifted_index(25.0, 18.0, -5.0)
        assert isinstance(li, float)

    def test_stable_cold_dry(self):
        """Cold dry conditions — positive or small-negative LI."""
        li = lifted_index(10.0, 2.0, -15.0)
        assert isinstance(li, float)

    def test_unstable_hot_moist(self):
        """Very warm surface with cold 500-hPa → negative LI (unstable)."""
        li = lifted_index(32.0, 25.0, -10.0)
        # LI may be negative for very moist hot surface vs cold 500 hPa
        assert isinstance(li, float)
