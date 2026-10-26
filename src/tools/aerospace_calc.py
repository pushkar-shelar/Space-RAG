"""
Deterministic Aerospace, Astrodynamics & Remote Sensing Calculation Utilities.
Used for precise, infallible physical calculations in Space RAG.
"""

from __future__ import annotations

import math
from typing import Any, Dict


# Physical Constants (WGS84 Earth Model)
EARTH_RADIUS_KM = 6378.137  # Equator radius
EARTH_MU = 398600.4418      # Standard gravitational parameter (km^3/s^2)
SPEED_OF_LIGHT_KM_S = 299792.458


def keplerian_orbital_period(altitude_km: float, mu: float = EARTH_MU, radius_km: float = EARTH_RADIUS_KM) -> Dict[str, Any]:
    """
    Calculate orbital period and velocity for a circular low-Earth or medium-Earth orbit.
    """
    semi_major_axis = radius_km + altitude_km
    if semi_major_axis <= 0:
        raise ValueError("Semi-major axis must be positive")

    period_sec = 2.0 * math.pi * math.sqrt((semi_major_axis ** 3) / mu)
    velocity_km_s = math.sqrt(mu / semi_major_axis)
    orbits_per_day = 86400.0 / period_sec

    return {
        "altitude_km": altitude_km,
        "semi_major_axis_km": round(semi_major_axis, 3),
        "period_seconds": round(period_sec, 2),
        "period_minutes": round(period_sec / 60.0, 2),
        "period_hours": round(period_sec / 3600.0, 3),
        "orbital_velocity_km_s": round(velocity_km_s, 4),
        "orbits_per_day": round(orbits_per_day, 2),
    }


def hohmann_transfer_delta_v(r1_alt_km: float, r2_alt_km: float, mu: float = EARTH_MU, radius_km: float = EARTH_RADIUS_KM) -> Dict[str, Any]:
    """
    Calculate the two-impulse Hohmann transfer delta-V between two circular orbits.
    """
    r1 = radius_km + r1_alt_km
    r2 = radius_km + r2_alt_km

    # Velocity in initial and final circular orbits
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)

    # Semi-major axis of transfer ellipse
    a_transfer = (r1 + r2) / 2.0

    # Velocities at periapsis and apoapsis of transfer orbit
    v_trans_1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_trans_2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

    # Delta-V burns
    dv1 = abs(v_trans_1 - v1)
    dv2 = abs(v2 - v_trans_2)
    total_dv = dv1 + dv2

    # Transfer duration (half orbital period of transfer ellipse)
    transfer_time_sec = math.pi * math.sqrt((a_transfer ** 3) / mu)

    return {
        "initial_altitude_km": r1_alt_km,
        "target_altitude_km": r2_alt_km,
        "burn_1_delta_v_km_s": round(dv1, 4),
        "burn_1_delta_v_m_s": round(dv1 * 1000.0, 2),
        "burn_2_delta_v_km_s": round(dv2, 4),
        "burn_2_delta_v_m_s": round(dv2 * 1000.0, 2),
        "total_delta_v_km_s": round(total_dv, 4),
        "total_delta_v_m_s": round(total_dv * 1000.0, 2),
        "transfer_time_minutes": round(transfer_time_sec / 60.0, 2),
        "transfer_time_hours": round(transfer_time_sec / 3600.0, 3),
    }


def free_space_path_loss(distance_km: float, frequency_ghz: float) -> Dict[str, Any]:
    """
    Calculate Free Space Path Loss (FSPL) in dB given distance and RF frequency.
    Formula: FSPL (dB) = 20*log10(d_km) + 20*log10(f_GHz) + 92.45
    """
    if distance_km <= 0 or frequency_ghz <= 0:
        raise ValueError("Distance and frequency must be positive.")

    fspl_db = 20.0 * math.log10(distance_km) + 20.0 * math.log10(frequency_ghz) + 92.4477

    return {
        "distance_km": distance_km,
        "frequency_ghz": frequency_ghz,
        "path_loss_db": round(fspl_db, 2),
    }


def sensor_ground_swath(altitude_km: float, cross_track_fov_deg: float) -> Dict[str, Any]:
    """
    Calculate optical or radar sensor cross-track ground swath width on Earth surface.
    """
    half_angle_rad = math.radians(cross_track_fov_deg / 2.0)
    swath_km = 2.0 * altitude_km * math.tan(half_angle_rad)

    return {
        "altitude_km": altitude_km,
        "fov_degrees": cross_track_fov_deg,
        "swath_width_km": round(swath_km, 2),
    }


def check_and_execute_aerospace_calc(query: str) -> Dict[str, Any] | None:
    """
    Inspect user query for deterministic aerospace calculations.
    Returns structured result dict or None if no calculation intent is matched.
    """
    import re
    q = query.lower()

    # 1. Hohmann Transfer Delta-V: e.g. "delta-v from 500 km to 75000 km"
    m_dv = re.search(r"(?:delta-?v|hohmann|transfer)\s+(?:from\s+)?(\d+(?:\.\d+)?)\s*km\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*km", q)
    if m_dv:
        alt1 = float(m_dv.group(1))
        alt2 = float(m_dv.group(2))
        res = hohmann_transfer_delta_v(alt1, alt2)
        summary = (
            f"Deterministic Astrodynamics Hohmann Transfer Calculation:\n"
            f"- Initial Orbit Altitude: {res['initial_altitude_km']} km\n"
            f"- Target Orbit Altitude: {res['target_altitude_km']} km\n"
            f"- Burn 1 Delta-V: {res['burn_1_delta_v_m_s']} m/s ({res['burn_1_delta_v_km_s']} km/s)\n"
            f"- Burn 2 Delta-V: {res['burn_2_delta_v_m_s']} m/s ({res['burn_2_delta_v_km_s']} km/s)\n"
            f"- Total Mission Delta-V Required: {res['total_delta_v_m_s']} m/s ({res['total_delta_v_km_s']} km/s)\n"
            f"- One-way Transfer Duration: {res['transfer_time_minutes']} minutes ({res['transfer_time_hours']} hours)"
        )
        return {"type": "hohmann_transfer", "data": res, "text": summary}

    # 2. Orbital Period / Velocity: e.g. "orbital period at 550 km" or "velocity at 550 km"
    m_orb = re.search(r"(?:orbital\s+period|velocity|speed|how\s+long\s+to\s+orbit)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*km", q)
    if not m_orb:
        m_orb = re.search(r"(\d+(?:\.\d+)?)\s*km\s+(?:altitude\s+)?(?:orbital\s+period|velocity|speed)", q)
    if m_orb:
        alt = float(m_orb.group(1))
        res = keplerian_orbital_period(alt)
        summary = (
            f"Deterministic Keplerian Orbit Calculation:\n"
            f"- Altitude: {res['altitude_km']} km\n"
            f"- Semi-Major Axis: {res['semi_major_axis_km']} km\n"
            f"- Orbital Period: {res['period_minutes']} minutes ({res['period_seconds']} seconds / {res['period_hours']} hours)\n"
            f"- Mean Circular Orbital Velocity: {res['orbital_velocity_km_s']} km/s ({round(res['orbital_velocity_km_s']*1000, 1)} m/s)\n"
            f"- Constellation Revisit / Orbits per Day: {res['orbits_per_day']} orbits/day"
        )
        return {"type": "keplerian_orbit", "data": res, "text": summary}

    # 3. Free Space Path Loss (FSPL): e.g. "path loss at 550 km and 118 ghz"
    m_fspl = re.search(r"(?:path\s+loss|fspl|attenuation)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*km\s+(?:and\s+|at\s+)?(\d+(?:\.\d+)?)\s*(?:ghz|mhz)", q)
    if m_fspl:
        dist = float(m_fspl.group(1))
        freq_raw = float(m_fspl.group(2))
        freq_ghz = freq_raw / 1000.0 if "mhz" in m_fspl.group(0) else freq_raw
        res = free_space_path_loss(dist, freq_ghz)
        summary = (
            f"Deterministic RF Link Budget Path Loss Calculation:\n"
            f"- Slant Range / Distance: {res['distance_km']} km\n"
            f"- Frequency: {res['frequency_ghz']} GHz\n"
            f"- Free Space Path Loss (FSPL): {res['path_loss_db']} dB"
        )
        return {"type": "path_loss", "data": res, "text": summary}

    # 4. Swath Width: e.g. "swath width at 550 km with 60 deg fov"
    m_swath = re.search(r"(?:swath|ground\s+track)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*km\s+(?:with\s+)?(\d+(?:\.\d+)?)\s*deg", q)
    if m_swath:
        alt = float(m_swath.group(1))
        fov = float(m_swath.group(2))
        res = sensor_ground_swath(alt, fov)
        summary = (
            f"Deterministic Remote Sensing Sensor Geometry:\n"
            f"- Orbital Altitude: {res['altitude_km']} km\n"
            f"- Cross-Track FOV: {res['fov_degrees']} degrees\n"
            f"- Ground Swath Footprint Width: {res['swath_width_km']} km"
        )
        return {"type": "swath_width", "data": res, "text": summary}

    return None

