"""Sensor model for TKID wiki estimates.

Wiki references:
- wiki/project.html
- wiki/theory.html
- wiki/noise-sources.html
- wiki/noise-johnson.html
- wiki/noise-tls.html
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import log, log10, pi, sqrt
from typing import Dict, Tuple
import numpy as np

K_B = 1.380649e-23
MU0 = 4.0e-7 * pi
J_PER_EV = 1.602176634e-19
N_A = 6.02214076e23
RHO_AU_KG_PER_M3 = 19300.0
M_AU_KG_PER_MOL = 0.19696657
HO163_HALF_LIFE_Y = 4570.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class SensorInputs:
    """Primary model inputs only (no derived fields)."""

    # Operating temperatures
    T0_K: float
    Tb_K: float

    # Geometry + thermal-capacity requirements (primary project knobs)
    heat_capacity_eV_per_mK: float
    ho_in_au_atomic_fraction: float
    ho_decay_energy_J: float
    kid_length_m: float
    kid_width_m: float
    membrane_margin_m: float
    leg_count: int
    leg_width_m: float
    cap_thickness_m: float
    membrane_thickness_m: float

    # Material properties used with geometry (independent inputs)
    cv_absorber_J_per_m3K: float
    kappa_leg_W_per_mK: float
    thermal_link_exponent_n: float

    # Simplified TLS model inputs
    tls_phi_asd_100hz_per_rtHz: float
    tls_beta: float

    # Resonator/electrical operating point
    f0_Hz: float
    Qi: float
    Qc: float
    tau_qp_s: float
    kinetic_inductance_fraction: float
    kid_trace_length_m: float
    kid_trace_width_m: float
    alpha_A: float
    alpha_phi: float
    beta_A: float
    beta_phi: float
    Tc_K: float
    pg_drive_dBm: float
    bifurcation_energy_scale_J: float
    pbif_typical_min_dBm: float
    pbif_typical_max_dBm: float
    thermal_energy_resolution_target_eV: float

    # Readout condition
    detuning_widths: float
    nep_sufficiency_percent: float
    # Optional second series KID / second island parameters
    T02_K: float = 0.0
    heat_capacity2_eV_per_mK: float = 0.0
    G2_W_per_K: float = 0.0
    alpha_A2: float = 0.0
    alpha_phi2: float = 0.0
    series_L2_H: float = 0.0
    series_R2_Ohm: float = 0.0
    feedback_heater_gain_W_per_rad: float = 0.0
    f_demod_Hz: float = 0.0


@dataclass(frozen=True)
class Version1SensorInputs(SensorInputs):
    """Primary project preset (version 1 configuration)."""

    T0_K: float = 0.040
    Tb_K: float = 0.020
    heat_capacity_eV_per_mK: float = 60.0
    ho_in_au_atomic_fraction: float = 0.00068
    ho_decay_energy_J: float = 4.5e-16
    kid_length_m: float = 220e-6
    kid_width_m: float = 220e-6
    membrane_margin_m: float = 20e-6
    leg_count: int = 4
    leg_width_m: float = 0.50e-6
    cap_thickness_m: float = 1.0e-6
    membrane_thickness_m: float = 1.0e-6
    cv_absorber_J_per_m3K: float = 0.075
    kappa_leg_W_per_mK: float = 1.5e-3
    thermal_link_exponent_n: float = 3.0
    tls_phi_asd_100hz_per_rtHz: float = 1.0e-6
    tls_beta: float = 0.5
    f0_Hz: float = 1.0e9
    Qi: float = 50000.0
    Qc: float = 50000.0
    tau_qp_s: float = 5.0e-8
    kinetic_inductance_fraction: float = 0.5
    kid_trace_length_m: float = 10.0e-3
    kid_trace_width_m: float = 2.0e-6
    alpha_A: float = 0.1
    alpha_phi: float = 140.0
    beta_A: float = 0.0
    beta_phi: float = 0.0
    Tc_K: float = 2.0
    pg_drive_dBm: float = -100.0
    bifurcation_energy_scale_J: float = 1.4323944878270582e-13
    pbif_typical_min_dBm: float = -95.0
    pbif_typical_max_dBm: float = -70.0
    thermal_energy_resolution_target_eV: float = 0.1
    detuning_widths: float = 0.1
    nep_sufficiency_percent: float = 10.0
    T02_K: float = 0.070
    heat_capacity2_eV_per_mK: float = 60.0
    G2_W_per_K: float = 2.1021957e-12
    alpha_A2: float = 0.1
    alpha_phi2: float = 140.0
    series_L2_H: float = 2.4404747e-07
    series_R2_Ohm: float = 0.03066791
    feedback_heater_gain_W_per_rad: float = 0.0
    f_demod_Hz: float = 0.0

@dataclass(frozen=True)
class Sensor:
    """Sensor model instantiated from a SensorInputs object.

    Basic inputs are geometry + heat-capacity requirements, plus independent
    electrical/material constants. Leg width is a basic geometry parameter.
    Count rate derives from absorber volume and Ho activity. G derives from C/tau,
    and leg length derives from G and leg geometry/material parameters.

    Notes:
    - Noise estimates are evaluated in phase direction at demodulated frequency 0 Hz.
    - Modulated-band frequency at this point is the carrier f0.
    - Zero detuning is currently assumed and stored as a project parameter.
    """
    inputs: SensorInputs

    def __getattr__(self, name: str):
        return getattr(self.inputs, name)

    @cached_property
    def detuning_Hz(self) -> float:
        """Readout detuning derived from resonator-width units (fr/Qr)."""
        return self.detuning_widths * (self.f0_Hz / self.Qr)

    @cached_property
    def x(self) -> float:
        """Dimensionless detuning, x = detuning_Hz / f0_Hz."""
        return self.detuning_Hz / self.f0_Hz

    @cached_property
    def au_number_density_per_m3(self) -> float:
        """Gold atomic number density (atoms/m^3)."""
        return (RHO_AU_KG_PER_M3 / M_AU_KG_PER_MOL) * N_A

    @cached_property
    def ho_number_density_per_m3(self) -> float:
        """Ho number density from chosen Ho/Au atomic fraction."""
        return self.ho_in_au_atomic_fraction * self.au_number_density_per_m3

    @cached_property
    def ho_decay_constant_per_s(self) -> float:
        """163Ho decay constant lambda in 1/s."""
        t_half_s = HO163_HALF_LIFE_Y * SECONDS_PER_YEAR
        return log(2.0) / t_half_s

    @cached_property
    def ho_activity_per_m3_Hz(self) -> float:
        """Derived volumetric activity from Ho number density."""
        return self.ho_decay_constant_per_s * self.ho_number_density_per_m3

    @cached_property
    def absorber_volume_m3(self) -> float:
        """Absorber volume implied by heat capacity and absorber material c_v."""
        return self.C_J_per_K / self.cv_absorber_J_per_m3K

    @cached_property
    def absorber_edge_m(self) -> float:
        """Cubic absorber edge from derived volume."""
        return self.absorber_volume_m3 ** (1.0 / 3.0)

    @cached_property
    def absorber_length_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def absorber_width_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def absorber_thickness_m(self) -> float:
        return self.absorber_edge_m

    @cached_property
    def membrane_length_m(self) -> float:
        """Derived membrane length from absorber + KID + margin."""
        return self.absorber_length_m + self.kid_length_m + self.membrane_margin_m

    @cached_property
    def membrane_width_m(self) -> float:
        """Derived membrane width from absorber + KID + margin."""
        return self.absorber_width_m + self.kid_width_m + self.membrane_margin_m

    @cached_property
    def membrane_span_m(self) -> float:
        """Characteristic span for leg scaling."""
        return max(self.membrane_length_m, self.membrane_width_m)

    @cached_property
    def leg_thickness_m(self) -> float:
        """Leg thickness derived from membrane thickness."""
        return self.membrane_thickness_m

    @cached_property
    def C_J_per_K(self) -> float:
        """Heat capacity converted from primary input heat_capacity_eV_per_mK."""
        return self.heat_capacity_eV_per_mK * 1.0e3 * J_PER_EV

    @cached_property
    def count_rate_Hz(self) -> float:
        """Derived count rate from Ho activity and absorber volume."""
        return self.ho_activity_per_m3_Hz * self.absorber_volume_m3

    @cached_property
    def tau_resolve_s(self) -> float:
        """Resolve window from pileup requirement: P ~ R * tau_resolve."""
        return self.nep_sufficient_time_s

    @cached_property
    def tau_th_s(self) -> float:
        """Thermal decay tau from thermal link and heat capacity."""
        return self.C_J_per_K / self.G_W_per_K

    @cached_property
    def G_W_per_K(self) -> float:
        """Thermal conductance set by operating-point temperature elevation."""
        if self.deltaT_abs_over_bath_setpoint_K <= 0.0:
            raise ValueError("T0_K must be greater than Tb_K to define positive thermal elevation")
        return self.P0_W / self.deltaT_abs_over_bath_setpoint_K

    @cached_property
    def deltaT_abs_over_bath_setpoint_K(self) -> float:
        """Operating-point elevation setpoint derived from T0 and Tb."""
        return self.T0_K - self.Tb_K

    @cached_property
    def deltaT_abs_over_bath_K(self) -> float:
        """Absorber/KID operating-point temperature elevation over bath."""
        return self.deltaT_abs_over_bath_setpoint_K

    @cached_property
    def tbath_from_link_K(self) -> float:
        """Bath temperature inferred from KID temperature minus thermal-link rise."""
        return self.T0_K - self.deltaT_abs_over_bath_K

    @cached_property
    def deltaT_event_full_absorption_K(self) -> float:
        """Island temperature step from a fully absorbed Ho decay event."""
        return self.ho_decay_energy_J / self.C_J_per_K

    @cached_property
    def C_eV_per_mK(self) -> float:
        """Island heat capacity in eV/mK (primary input value)."""
        return self.heat_capacity_eV_per_mK

    @cached_property
    def C_ho_eV_per_mK(self) -> float:
        """Ho absorber heat capacity in eV/mK."""
        return self.heat_capacity_eV_per_mK

    @cached_property
    def ho_decay_energy_eV(self) -> float:
        """Ho decay event energy in eV."""
        return self.ho_decay_energy_J / J_PER_EV

    @cached_property
    def leg_length_m(self) -> float:
        """Derived leg length from G and leg geometry/material.

        Rearranged from G = N * kappa * (w * t / L):
        L = N * kappa * w * t / G
        """
        return (
            self.leg_count
            * self.kappa_leg_W_per_mK
            * self.leg_width_m
            * self.leg_thickness_m
            / self.G_W_per_K
        )

    @cached_property
    def tau_res_s(self) -> float:
        return self.Qr / (pi * self.f0_Hz)

    @cached_property
    def L_geo_H(self) -> float:
        """Approximate geometric inductance from trace length/width.

        Thin straight-trace approximation:
        L ~ mu0 * l * (ln(2l/w) + 0.5)
        """
        l = self.kid_trace_length_m
        w = self.kid_trace_width_m
        ratio = max((2.0 * l) / w, 1.000001)
        return MU0 * l * (log(ratio) + 0.5)

    @cached_property
    def L_total_H(self) -> float:
        """Total inductance from geometric inductance and kinetic fraction."""
        kfrac = self.kinetic_inductance_fraction
        if not (0.0 <= kfrac < 1.0):
            raise ValueError("kinetic_inductance_fraction must be in [0,1)")
        return self.L_geo_H / (1.0 - kfrac)

    @cached_property
    def L_series_total_H(self) -> float:
        """Total series inductance including optional second KID."""
        return self.L_total_H + max(self.series_L2_H, 0.0)

    @cached_property
    def C_res_F(self) -> float:
        """Resonator capacitance from physical f0 and active series inductance."""
        w0 = 2.0 * pi * self.f0_Hz
        lsum = self.L_series_total_H
        if lsum <= 0.0:
            return float("nan")
        return 1.0 / (w0 * w0 * lsum)

    @cached_property
    def Z0_res_Ohm(self) -> float:
        """Nominal resonator characteristic impedance at f0."""
        w0 = 2.0 * pi * self.f0_Hz
        return w0 * self.L_series_total_H

    @cached_property
    def R0_Ohm(self) -> float:
        """Series-equivalent resonator resistance from total L and Qi."""
        return self.R_series_total_Ohm

    @cached_property
    def R1_series_Ohm(self) -> float:
        """Series loss assigned to KID1 from baseline Qi and L1."""
        if self.Qi <= 0.0:
            return float("nan")
        return (2.0 * pi * self.f0_Hz * self.L_total_H) / self.Qi

    @cached_property
    def R_series_total_Ohm(self) -> float:
        """Total series loss from KID1 plus optional KID2 loss."""
        return self.R1_series_Ohm + max(self.series_R2_Ohm, 0.0)

    @cached_property
    def Qi_eff(self) -> float:
        """Internal Q from total series L/R at effective resonance."""
        rtot = self.R_series_total_Ohm
        if rtot <= 0.0 or self.f0_Hz <= 0.0:
            return float("nan")
        return (2.0 * pi * self.f0_Hz * self.L_series_total_H) / rtot

    @cached_property
    def Qr(self) -> float:
        """Loaded resonator Q derived from internal and coupling Q values."""
        if self.Qi_eff <= 0.0 or self.Qc <= 0.0:
            return float("nan")
        return 1.0 / ((1.0 / self.Qi_eff) + (1.0 / self.Qc))

    @cached_property
    def phonon_power_rms_W(self) -> float:
        """Phonon TFN ASD with two-temperature weak-link correction."""
        return self.phonon_power_asd_device_W_per_rtHz

    @cached_property
    def phonon_power_asd_device_W_per_rtHz(self) -> float:
        """Phonon TFN ASD: sqrt(4 k_B T0^2 G F_link(T0,Tb,n))."""
        return self._phonon_power_asd_for(self.T0_K, self.G_W_per_K)

    def _phonon_power_asd_for(self, temperature_K: float, conductance_W_per_K: float) -> float:
        n = self.thermal_link_exponent_n
        if n <= -1.0:
            raise ValueError("thermal_link_exponent_n must be > -1")
        if temperature_K <= 0.0 or self.Tb_K <= 0.0:
            raise ValueError("thermal node temperature and Tb_K must be > 0")
        if conductance_W_per_K <= 0.0:
            return 0.0
        r = self.Tb_K / temperature_K
        denom = 1.0 - (r ** (n + 1.0))
        if abs(denom) < 1.0e-15:
            f_link = 1.0
        else:
            num = 1.0 - (r ** (2.0 * n + 3.0))
            f_link = ((n + 1.0) / (2.0 * n + 3.0)) * (num / denom)
        return sqrt(4.0 * K_B * (temperature_K**2) * conductance_W_per_K * f_link)

    @cached_property
    def johnson_voltage_rms_V(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.R0_Ohm)

    @cached_property
    def johnson_sv_V2_per_Hz(self) -> float:
        return 4.0 * K_B * self.T0_K * self.R0_Ohm

    @cached_property
    def me_electronic(self) -> float:
        return sqrt(self.eqp_J / (K_B * self.T0_K))

    @cached_property
    def delta_J(self) -> float:
        """BCS gap: Delta = 1.764 k_B Tc."""
        return 1.764 * K_B * self.Tc_K

    @cached_property
    def eqp_J(self) -> float:
        """Paper convention: Eqp ~ Delta."""
        return self.delta_J

    @cached_property
    def nj_scale(self) -> float:
        return sqrt(16.0 * K_B * self.T0_K / self.P0_W)

    @cached_property
    def nj_thermal_scale(self) -> float:
        return sqrt(4.0 * K_B * self.T0_K * self.P0_W)

    @cached_property
    def T02_eff_K(self) -> float:
        """Effective KID2 operating temperature (falls back to T0)."""
        return self.T02_K if self.T02_K > 0.0 else self.T0_K

    @cached_property
    def P1_W(self) -> float:
        r1 = max(self.R1_series_Ohm, 0.0)
        r2 = max(self.series_R2_Ohm, 0.0)
        rsum = r1 + r2
        if rsum <= 0.0:
            return self.P0_W
        return self.P0_W * (r1 / rsum)

    @cached_property
    def P2_W(self) -> float:
        r1 = max(self.R1_series_Ohm, 0.0)
        r2 = max(self.series_R2_Ohm, 0.0)
        rsum = r1 + r2
        if rsum <= 0.0:
            return 0.0
        return self.P0_W * (r2 / rsum)

    @cached_property
    def R1_fraction(self) -> float:
        r1 = max(self.R1_series_Ohm, 0.0)
        r2 = max(self.series_R2_Ohm, 0.0)
        rsum = r1 + r2
        if rsum <= 0.0:
            return 1.0
        return r1 / rsum

    @cached_property
    def R2_fraction(self) -> float:
        r1 = max(self.R1_series_Ohm, 0.0)
        r2 = max(self.series_R2_Ohm, 0.0)
        rsum = r1 + r2
        if rsum <= 0.0:
            return 0.0
        return r2 / rsum

    @cached_property
    def nj1_scale(self) -> float:
        if self.R1_fraction <= 0.0:
            return 0.0
        return self.nj_scale * sqrt(self.R1_fraction)

    @cached_property
    def nj2_scale(self) -> float:
        if self.R2_fraction <= 0.0:
            return 0.0
        return self.nj_scale * sqrt(self.R2_fraction * (self.T02_eff_K / self.T0_K))

    @cached_property
    def nj1_thermal_scale(self) -> float:
        if self.P1_W <= 0.0:
            return 0.0
        return sqrt(4.0 * K_B * self.T0_K * self.P1_W)

    @cached_property
    def nj2_thermal_scale(self) -> float:
        if self.P2_W <= 0.0:
            return 0.0
        return sqrt(4.0 * K_B * self.T02_eff_K * self.P2_W)

    @cached_property
    def heater2_dc_power_W(self) -> float:
        """DC heater power needed to hold island 2 at T02."""
        if not self.second_kid_active or self.G2_W_per_K <= 0.0:
            return 0.0
        return (self.G2_W_per_K * (self.T02_eff_K - self.Tb_K)) - self.P2_W

    @cached_property
    def heater2_dc_power_dBm(self) -> float:
        if self.heater2_dc_power_W <= 0.0:
            return float("nan")
        return 10.0 * log10(self.heater2_dc_power_W / 1.0e-3)

    @cached_property
    def tau_target_from_rate_s(self) -> float:
        """Symbol used in outputs to mirror rate-derived tau target."""
        return 5.0e-4 / self.count_rate_Hz

    @cached_property
    def pileup_probability_max(self) -> float:
        """Per-event rejection probability for a +/-tau coincidence veto window.

        Uses tau = nep_sufficient_time_s and Poisson arrivals with rate R.
        An event is rejected if any other event occurs within +/-tau, so:
            P_reject = 1 - exp(-2 R tau)
        This rejects both pulses in a colliding pair.
        """
        arg = -2.0 * self.count_rate_Hz * self.nep_sufficient_time_s
        return 1.0 - float(np.exp(arg))

    def _phase_nep_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Phase-total NEP spectrum used for NEP sufficiency calculations."""
        eigs = np.array(self.mt_eigenvalues, dtype=complex)
        max_mode_rate_per_s = float(np.max(np.abs(eigs)))
        f_min_hz = 0.1
        f_max_hz = max(f_min_hz * 10.0, 10.0 * max_mode_rate_per_s / (2.0 * pi))
        freqs_hz = np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), 1000)

        asd_phase_johnson = np.zeros_like(freqs_hz)
        asd_phase_phonon = np.zeros_like(freqs_hz)
        asd_phase_tls = np.zeros_like(freqs_hz)
        asd_phase_electronic = np.zeros_like(freqs_hz)
        phase_resp = np.zeros_like(freqs_hz)

        for i, f_hz in enumerate(freqs_hz):
            y_j_a1 = self._propagate_noise_vector(self.n_johnson_A_1(), float(f_hz))
            y_j_phi1 = self._propagate_noise_vector(self.n_johnson_phi_1(), float(f_hz))
            y_j_a2 = self._propagate_noise_vector(self.n_johnson_A_2(), float(f_hz))
            y_j_phi2 = self._propagate_noise_vector(self.n_johnson_phi_2(), float(f_hz))
            y_ph1 = self._propagate_noise_vector(self.n_phonon_1(), float(f_hz))
            y_ph2 = self._propagate_noise_vector(self.n_phonon_2(), float(f_hz))
            y_tls = self._propagate_noise_vector(self.n_tls_phi_at_hz(float(f_hz)), float(f_hz))
            y_e_a1 = self._propagate_noise_vector(self.n_electronic_A_1(), float(f_hz))
            y_e_phi1 = self._propagate_noise_vector(self.n_electronic_phi_1(), float(f_hz))
            y_e_a2 = self._propagate_noise_vector(self.n_electronic_A_2(), float(f_hz))
            y_e_phi2 = self._propagate_noise_vector(self.n_electronic_phi_2(), float(f_hz))

            asd_phase_johnson[i] = np.sqrt(abs(y_j_a1[1]) ** 2 + abs(y_j_phi1[1]) ** 2 + abs(y_j_a2[1]) ** 2 + abs(y_j_phi2[1]) ** 2)
            asd_phase_phonon[i] = np.sqrt(abs(y_ph1[1]) ** 2 + abs(y_ph2[1]) ** 2)
            asd_phase_tls[i] = abs(y_tls[1])
            asd_phase_electronic[i] = np.sqrt(abs(y_e_a1[1]) ** 2 + abs(y_e_phi1[1]) ** 2 + abs(y_e_a2[1]) ** 2 + abs(y_e_phi2[1]) ** 2)
            phase_resp[i] = self.phase_responsivity_mag_rad_per_W_at_hz(float(f_hz))

        asd_phase_total = np.sqrt(asd_phase_johnson**2 + asd_phase_phonon**2 + asd_phase_tls**2 + asd_phase_electronic**2)
        nep_phase_total = np.where(phase_resp > 0.0, asd_phase_total / phase_resp, np.nan)
        return freqs_hz, nep_phase_total

    @staticmethod
    def _first_sufficiency_frequency_hz(
        f_hz: np.ndarray, nep_w_per_rthz: np.ndarray, sufficiency_percent: float
    ) -> float:
        """First frequency (high->low integration) within sufficiency_percent of full sigma."""
        f = np.asarray(f_hz, dtype=float)
        nep = np.asarray(nep_w_per_rthz, dtype=float)
        valid = np.isfinite(f) & np.isfinite(nep) & (f > 0.0) & (nep > 0.0)
        if np.count_nonzero(valid) < 2:
            return float("nan")
        f = f[valid]
        nep = nep[valid]
        order = np.argsort(f)
        f = f[order]
        nep = nep[order]
        if np.any(np.diff(f) <= 0.0):
            return float("nan")

        # Cumulative information from the high-frequency end down to each f.
        integrand = 4.0 / (nep * nep)
        df = np.diff(f)
        trap = 0.5 * (integrand[:-1] + integrand[1:]) * df
        inv_sigma2_prefix = np.zeros_like(f)
        inv_sigma2_prefix[1:] = np.cumsum(trap)
        inv_sigma2_full = float(inv_sigma2_prefix[-1])
        # Numerical guard: high->low cumulative information must be >= 0.
        inv_sigma2_cum = np.maximum(inv_sigma2_full - inv_sigma2_prefix, 0.0)
        if inv_sigma2_full <= 0.0 or not np.isfinite(inv_sigma2_full):
            return float("nan")

        sigma_full = 1.0 / sqrt(inv_sigma2_full)
        frac = max(0.0, float(sufficiency_percent)) / 100.0
        sigma_target = (1.0 + frac) * sigma_full
        sigma_cum = np.full_like(inv_sigma2_cum, np.inf, dtype=float)
        pos = inv_sigma2_cum > 0.0
        sigma_cum[pos] = 1.0 / np.sqrt(inv_sigma2_cum[pos])
        hit = sigma_cum <= sigma_target
        idxs = np.where(hit)[0]
        if idxs.size == 0:
            return float("nan")
        # First crossing while scanning from high to low.
        return float(f[idxs[-1]])

    @cached_property
    def nep_sufficient_frequency_hz(self) -> float:
        """First low->high frequency where cumulative NEP sigma is sufficiently close."""
        freqs_hz, nep_phase_total = self._phase_nep_spectrum()
        return self._first_sufficiency_frequency_hz(freqs_hz, nep_phase_total, self.nep_sufficiency_percent)

    @cached_property
    def nep_sufficient_time_s(self) -> float:
        """Characteristic window time from NEP-sufficient frequency."""
        f = self.nep_sufficient_frequency_hz
        if not np.isfinite(f) or f <= 0.0:
            return float("nan")
        return 1.0 / f

    @cached_property
    def tau_error_fraction(self) -> float:
        """Difference between solved tau and target tau (currently identical)."""
        if self.tau_target_from_rate_s == 0.0:
            return 0.0
        return (self.tau_th_s - self.tau_target_from_rate_s) / self.tau_target_from_rate_s

    @cached_property
    def tau_ratio_res_over_th(self) -> float:
        return self.tau_res_s / self.tau_th_s

    @cached_property
    def core_rule1_left_ratio(self) -> float:
        """Rule 1 left ratio: tau_qp / tau_res (should be << 1)."""
        return self.tau_qp_s / self.tau_res_s

    @cached_property
    def core_rule1_right_ratio(self) -> float:
        """Rule 1 right ratio: tau_res / tau_th (should be << 1)."""
        return self.tau_res_s / self.tau_th_s

    @cached_property
    def core_rule1_ok(self) -> bool:
        """Practical check for '<<' using 0.1 threshold on both ratios."""
        return (self.core_rule1_left_ratio < 0.1) and (self.core_rule1_right_ratio < 0.1)

    @cached_property
    def core_rule2_ratio(self) -> float:
        """Rule 2 ratio: tau_res / tau_th, must be <= 1."""
        return self.tau_res_s / self.tau_th_s

    @cached_property
    def core_rule2_ok(self) -> bool:
        return self.core_rule2_ratio <= 1.0

    @cached_property
    def sf_over_f0sq_tls_model(self) -> float:
        return self.sf_over_f0sq_tls()

    def n_phonon_1(self) -> Tuple[complex, complex, complex, complex]:
        return (0.0 + 0.0j, 0.0 + 0.0j, self.phonon_power_asd_device_W_per_rtHz + 0.0j, 0.0 + 0.0j)

    def n_phonon_2(self) -> Tuple[complex, complex, complex, complex]:
        if not self.second_kid_active or self.G2_W_per_K <= 0.0:
            return (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
        n2 = self._phonon_power_asd_for(self.T02_eff_K, self.G2_W_per_K)
        return (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, n2 + 0.0j)

    def n_phonon(self) -> Tuple[complex, complex, complex, complex]:
        a = self.n_phonon_1()
        b = self.n_phonon_2()
        return tuple(ai + bi for ai, bi in zip(a, b))

    def n_johnson_A_1(self) -> Tuple[complex, complex, complex, complex]:
        return (self.nj1_scale + 0.0j, 0.0 + 0.0j, -self.nj1_thermal_scale + 0.0j, 0.0 + 0.0j)

    def n_johnson_A_2(self) -> Tuple[complex, complex, complex, complex]:
        if not self.second_kid_active:
            return (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
        return (self.nj2_scale + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -self.nj2_thermal_scale + 0.0j)

    def n_johnson_A(self) -> Tuple[complex, complex, complex, complex]:
        a = self.n_johnson_A_1()
        b = self.n_johnson_A_2()
        return tuple(ai + bi for ai, bi in zip(a, b))

    def n_johnson_phi_1(self) -> Tuple[complex, complex, complex, complex]:
        return (0.0 + 0.0j, 1j * self.nj1_scale, 0.0 + 0.0j, 0.0 + 0.0j)

    def n_johnson_phi_2(self) -> Tuple[complex, complex, complex, complex]:
        if not self.second_kid_active:
            return (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
        return (0.0 + 0.0j, 1j * self.nj2_scale, 0.0 + 0.0j, 0.0 + 0.0j)

    def n_johnson_phi(self) -> Tuple[complex, complex, complex, complex]:
        a = self.n_johnson_phi_1()
        b = self.n_johnson_phi_2()
        return tuple(ai + bi for ai, bi in zip(a, b))

    def _propagate_noise_vector(self, n_vec: Tuple[complex, complex, complex, complex], f_hz: float) -> np.ndarray:
        """Propagate one source vector through Y = M^{-1} N."""
        m = self.m_matrix_array(f_hz)
        n = np.array(n_vec, dtype=complex)
        return np.linalg.solve(m, n)

    @cached_property
    def y_johnson_A(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_A(), self.f_demod_Hz)

    @cached_property
    def y_johnson_phi(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_phi(), self.f_demod_Hz)

    @cached_property
    def y_johnson_A_1(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_A_1(), self.f_demod_Hz)

    @cached_property
    def y_johnson_phi_1(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_phi_1(), self.f_demod_Hz)

    @cached_property
    def y_johnson_A_2(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_A_2(), self.f_demod_Hz)

    @cached_property
    def y_johnson_phi_2(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_johnson_phi_2(), self.f_demod_Hz)

    @cached_property
    def y_phonon(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_phonon(), self.f_demod_Hz)

    @cached_property
    def y_phonon_1(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_phonon_1(), self.f_demod_Hz)

    @cached_property
    def y_phonon_2(self) -> np.ndarray:
        return self._propagate_noise_vector(self.n_phonon_2(), self.f_demod_Hz)

    @cached_property
    def y_tls_phi(self) -> np.ndarray:
        """TLS-driven output vector from phase-like TLS source at f_demod."""
        return self._propagate_noise_vector(self.n_tls_phi_at_hz(self.f_demod_Hz), self.f_demod_Hz)

    @cached_property
    def y_electronic_A(self) -> np.ndarray:
        """Electronic (generation-like) A-noise output vector at f_demod."""
        return self._propagate_noise_vector(self.n_electronic_A(), self.f_demod_Hz)

    @cached_property
    def y_electronic_phi(self) -> np.ndarray:
        """Electronic (generation-like) phi-noise output vector at f_demod."""
        return self._propagate_noise_vector(self.n_electronic_phi(), self.f_demod_Hz)

    def phase_responsivity_complex_rad_per_W_at_hz(self, f_hz: float) -> complex:
        """Complex phase responsivity to power source: (M^-1)_{phi,power}."""
        m = self.m_matrix_array(f_hz)
        e_power = np.array((0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j), dtype=complex)
        y_unit_power = np.linalg.solve(m, e_power)
        return complex(y_unit_power[1])

    def phase_responsivity_mag_rad_per_W_at_hz(self, f_hz: float) -> float:
        """Magnitude of phase responsivity to power source."""
        return float(abs(self.phase_responsivity_complex_rad_per_W_at_hz(f_hz)))

    @cached_property
    def phase_responsivity_rad_per_W(self) -> float:
        """Phase responsivity magnitude at f_demod."""
        return self.phase_responsivity_mag_rad_per_W_at_hz(self.f_demod_Hz)

    def nep_from_phase_asd_W_per_rtHz(self, asd_phi_per_rtHz: float, f_hz: float | None = None) -> float:
        """Convert phase ASD to NEP using |dphi/dP| at selected frequency."""
        f_eval = self.f_demod_Hz if f_hz is None else f_hz
        resp = self.phase_responsivity_mag_rad_per_W_at_hz(f_eval)
        if resp == 0.0:
            return float("nan")
        return asd_phi_per_rtHz / resp

    def sigma_energy_from_nep_spectrum_J(self, f_hz: np.ndarray, nep_W_per_rtHz: np.ndarray) -> float:
        """Estimate calorimetric RMS energy resolution from NEP(f), in joules.

        Uses the standard optimal-filter NEP relation:
            sigma_E = ( integral(4 / NEP(f)^2 df) )^(-1/2)
        evaluated with trapezoidal integration over arbitrary frequency spacing.
        This works for linearly or logarithmically spaced frequency arrays.
        """
        f = np.asarray(f_hz, dtype=float)
        nep = np.asarray(nep_W_per_rtHz, dtype=float)
        if f.ndim != 1 or nep.ndim != 1 or f.size != nep.size:
            raise ValueError("f_hz and nep_W_per_rtHz must be 1D arrays with equal length")
        if f.size < 2:
            raise ValueError("at least two frequency samples are required")
        if np.any(~np.isfinite(f)) or np.any(~np.isfinite(nep)):
            raise ValueError("inputs must be finite")
        if np.any(f <= 0.0):
            raise ValueError("frequencies must be > 0")
        if np.any(np.diff(f) <= 0.0):
            raise ValueError("frequencies must be strictly increasing")
        if np.any(nep <= 0.0):
            raise ValueError("NEP values must be > 0")
        inv_sigma2_integrand = 4.0 / (nep * nep)
        inv_sigma2 = float(np.trapezoid(inv_sigma2_integrand, x=f))
        if inv_sigma2 <= 0.0 or not np.isfinite(inv_sigma2):
            return float("nan")
        return 1.0 / sqrt(inv_sigma2)

    def sigma_energy_from_nep_spectrum_eV(self, f_hz: np.ndarray, nep_W_per_rtHz: np.ndarray) -> float:
        """Energy RMS sigma_E in eV from integrated NEP spectrum."""
        return self.sigma_energy_from_nep_spectrum_J(f_hz, nep_W_per_rtHz) / J_PER_EV

    @cached_property
    def sphi_johnson_full_per_hz(self) -> float:
        """Johnson phase-noise PSD from full matrix model at f_demod."""
        ys = (
            self.y_johnson_A_1,
            self.y_johnson_phi_1,
            self.y_johnson_A_2,
            self.y_johnson_phi_2,
        )
        return float(sum(abs(y[1]) ** 2 for y in ys))

    @cached_property
    def asd_phi_johnson_full_per_rtHz(self) -> float:
        """Johnson phase-noise ASD from full matrix model at f_demod."""
        return sqrt(self.sphi_johnson_full_per_hz)

    @cached_property
    def sphi_tls_per_hz(self) -> float:
        """TLS phase-noise PSD from full matrix model at f_demod."""
        return float(abs(self.y_tls_phi[1]) ** 2)

    @cached_property
    def asd_phi_tls_per_rtHz(self) -> float:
        """TLS phase-noise ASD from full matrix model at f_demod."""
        return sqrt(self.sphi_tls_per_hz)

    @cached_property
    def sphi_electronic_per_hz(self) -> float:
        """Electronic (generation-like) phase-noise PSD at f_demod."""
        ys = (
            self._propagate_noise_vector(self.n_electronic_A_1(), self.f_demod_Hz),
            self._propagate_noise_vector(self.n_electronic_phi_1(), self.f_demod_Hz),
            self._propagate_noise_vector(self.n_electronic_A_2(), self.f_demod_Hz),
            self._propagate_noise_vector(self.n_electronic_phi_2(), self.f_demod_Hz),
        )
        return float(sum(abs(y[1]) ** 2 for y in ys))

    @cached_property
    def asd_phi_electronic_per_rtHz(self) -> float:
        """Electronic (generation-like) phase-noise ASD at f_demod."""
        return sqrt(self.sphi_electronic_per_hz)

    @cached_property
    def sphi_total_per_hz(self) -> float:
        """Total phase-noise PSD from Johnson, TLS, phonon, and electronic sources."""
        return (
            self.sphi_johnson_full_per_hz
            + self.sphi_tls_per_hz
            + (self.asd_phi_phonon_full_per_rtHz**2)
            + self.sphi_electronic_per_hz
        )

    @cached_property
    def asd_phi_total_per_rtHz(self) -> float:
        """Total phase-noise ASD from quadrature sum of modeled sources."""
        return sqrt(self.sphi_total_per_hz)

    @cached_property
    def nep_phi_johnson_W_per_rtHz(self) -> float:
        return self.nep_from_phase_asd_W_per_rtHz(self.asd_phi_johnson_full_per_rtHz)

    @cached_property
    def nep_phi_tls_W_per_rtHz(self) -> float:
        return self.nep_from_phase_asd_W_per_rtHz(self.asd_phi_tls_per_rtHz)

    @cached_property
    def nep_phi_phonon_W_per_rtHz(self) -> float:
        return self.nep_from_phase_asd_W_per_rtHz(self.asd_phi_phonon_full_per_rtHz)

    @cached_property
    def nep_phi_electronic_W_per_rtHz(self) -> float:
        return self.nep_from_phase_asd_W_per_rtHz(self.asd_phi_electronic_per_rtHz)

    @cached_property
    def nep_phi_total_W_per_rtHz(self) -> float:
        return self.nep_from_phase_asd_W_per_rtHz(self.asd_phi_total_per_rtHz)

    @cached_property
    def nep_phi_phonon_0hz_W_per_rtHz(self) -> float:
        """Phase-direction phonon NEP evaluated at 0 Hz."""
        y_ph1 = self._propagate_noise_vector(self.n_phonon_1(), 0.0)
        y_ph2 = self._propagate_noise_vector(self.n_phonon_2(), 0.0)
        asd_phi_ph = float(sqrt(abs(y_ph1[1]) ** 2 + abs(y_ph2[1]) ** 2))
        resp0 = self.phase_responsivity_mag_rad_per_W_at_hz(0.0)
        if resp0 == 0.0:
            return float("nan")
        return asd_phi_ph / resp0

    @cached_property
    def nep_phi_total_0hz_W_per_rtHz(self) -> float:
        """Phase-direction total NEP evaluated at 0 Hz."""
        y_j_a1 = self._propagate_noise_vector(self.n_johnson_A_1(), 0.0)
        y_j_p1 = self._propagate_noise_vector(self.n_johnson_phi_1(), 0.0)
        y_j_a2 = self._propagate_noise_vector(self.n_johnson_A_2(), 0.0)
        y_j_p2 = self._propagate_noise_vector(self.n_johnson_phi_2(), 0.0)
        y_ph1 = self._propagate_noise_vector(self.n_phonon_1(), 0.0)
        y_ph2 = self._propagate_noise_vector(self.n_phonon_2(), 0.0)
        y_tls = self._propagate_noise_vector(self.n_tls_phi_at_hz(1.0), 0.0)
        y_e_a1 = self._propagate_noise_vector(self.n_electronic_A_1(), 0.0)
        y_e_p1 = self._propagate_noise_vector(self.n_electronic_phi_1(), 0.0)
        y_e_a2 = self._propagate_noise_vector(self.n_electronic_A_2(), 0.0)
        y_e_p2 = self._propagate_noise_vector(self.n_electronic_phi_2(), 0.0)
        asd_j = sqrt(abs(y_j_a1[1]) ** 2 + abs(y_j_p1[1]) ** 2 + abs(y_j_a2[1]) ** 2 + abs(y_j_p2[1]) ** 2)
        asd_ph = sqrt(abs(y_ph1[1]) ** 2 + abs(y_ph2[1]) ** 2)
        asd_tls = abs(y_tls[1])
        asd_e = sqrt(abs(y_e_a1[1]) ** 2 + abs(y_e_p1[1]) ** 2 + abs(y_e_a2[1]) ** 2 + abs(y_e_p2[1]) ** 2)
        asd_total = sqrt(asd_j**2 + asd_ph**2 + asd_tls**2 + asd_e**2)
        resp0 = self.phase_responsivity_mag_rad_per_W_at_hz(0.0)
        if resp0 == 0.0:
            return float("nan")
        return float(asd_total / resp0)

    @cached_property
    def nep_phi_0hz_over_phonon_ratio(self) -> float:
        """Ratio NEP_total(0 Hz) / NEP_phonon(0 Hz)."""
        denom = self.nep_phi_phonon_0hz_W_per_rtHz
        if denom == 0.0:
            return float("nan")
        return self.nep_phi_total_0hz_W_per_rtHz / denom

    @cached_property
    def dphi_df_detuning_per_hz(self) -> float:
        """Approximate local phase slope including detuning dependence."""
        return (4.0 * self.Qr / self.f0_Hz) / (1.0 + (2.0 * self.Qr * self.x) ** 2)

    @cached_property
    def dfr_dT_Hz_per_K(self) -> float:
        """Resonance-frequency thermal slope from alpha_phi definition."""
        # From alpha_phi = -2 Qi (T/fr) (dfr/dT), with fr ~ f0 at operating point.
        return -(self.alpha_phi * self.f0_Hz) / (2.0 * self.Qi_eff * self.T0_K)

    @cached_property
    def dT_dE_K_per_J(self) -> float:
        """Island temperature rise per absorbed energy."""
        return 1.0 / self.C_J_per_K

    @cached_property
    def dphi_dE_rad_per_J(self) -> float:
        """Simple chain estimate: dphi/dE = (dphi/df)(dfr/dT)(dT/dE)."""
        return self.dphi_df_detuning_per_hz * self.dfr_dT_Hz_per_K * self.dT_dE_K_per_J

    @cached_property
    def deltafr_event_Hz(self) -> float:
        """Resonance-frequency shift for one absorbed Ho event."""
        return self.dfr_dT_Hz_per_K * self.deltaT_event_full_absorption_K

    @cached_property
    def deltaphi_event_rad(self) -> float:
        """Phase shift estimate for one absorbed Ho event."""
        return self.dphi_dE_rad_per_J * self.ho_decay_energy_J

    @cached_property
    def phonon_temp_asd_device_K_per_rtHz(self) -> float:
        """Low-frequency island temperature ASD from phonon power ASD and G."""
        return self.phonon_power_asd_device_W_per_rtHz / self.G_W_per_K

    @cached_property
    def phonon_energy_asd_device_J_per_rtHz(self) -> float:
        """Equivalent low-frequency energy ASD from temperature ASD and C."""
        return self.C_J_per_K * self.phonon_temp_asd_device_K_per_rtHz

    @cached_property
    def asd_phi_phonon_simple_per_rtHz(self) -> float:
        """Simple low-frequency phase ASD from phonon energy ASD and dphi/dE."""
        return abs(self.dphi_dE_rad_per_J) * self.phonon_energy_asd_device_J_per_rtHz

    @cached_property
    def thermal_energy_fluct_rms_J(self) -> float:
        """Simple island thermal-energy fluctuation scale: sqrt(k_B T0^2 C)."""
        return sqrt(K_B * (self.T0_K**2) * self.C_J_per_K)

    @cached_property
    def thermal_energy_fluct_rms_eV(self) -> float:
        """Island thermal-energy fluctuation scale in eV."""
        return self.thermal_energy_fluct_rms_J / J_PER_EV

    @cached_property
    def sf_over_f0sq_johnson_full(self) -> float:
        """General conversion using detuning-dependent local phase slope."""
        slope = self.dphi_df_detuning_per_hz
        if slope == 0.0:
            return float("nan")
        return self.sphi_johnson_full_per_hz / ((self.f0_Hz**2) * (slope**2))

    @cached_property
    def sf_over_f0sq_johnson_simple(self) -> float:
        """Simplified on-resonance-style conversion with 4Qr slope."""
        return self.sphi_johnson_full_per_hz / ((4.0 * self.Qr) ** 2)

    @cached_property
    def m_phonon_over_johnson_phi(self) -> float:
        """Phase-quadrature ASD ratio at f_demod: phonon / Johnson-phi."""
        denom = abs(self.y_johnson_phi[1])
        if denom == 0.0:
            return float("nan")
        return float(abs(self.y_phonon[1]) / denom)

    def fractional_frequency_psd_from_phase(self, s_phi_per_hz: float) -> float:
        return s_phi_per_hz / (4.0 * self.Qr) ** 2

    def sf_over_f0sq_tls(self) -> float:
        """TLS fractional frequency-noise PSD at model reference offset (1 Hz)."""
        return self.sf_over_f0sq_tls_at_hz(1.0)

    def sf_over_f0sq_tls_at_hz(self, nu_hz: float) -> float:
        """TLS fractional frequency-noise PSD from phase ASD anchor at 100 Hz."""
        if nu_hz <= 0.0:
            raise ValueError("nu_hz must be > 0")
        asd_phi_nu = self.tls_phi_asd_at_hz_per_rtHz(nu_hz)
        gain = abs(self.f0_Hz * self.dphi_df_detuning_per_hz)
        if gain == 0.0:
            return float("nan")
        return (asd_phi_nu / gain) ** 2

    def tls_phi_asd_at_hz_per_rtHz(self, nu_hz: float) -> float:
        """TLS phase ASD power-law anchored at 100 Hz."""
        if nu_hz <= 0.0:
            raise ValueError("nu_hz must be > 0")
        return self.tls_phi_asd_100hz_per_rtHz * ((nu_hz / 100.0) ** (-self.tls_beta / 2.0))

    @cached_property
    def I0_rms_A(self) -> float:
        """RMS current from resonator dissipation relation P0 = I0^2 R0."""
        return sqrt(self.P0_W / self.R0_Ohm)

    @cached_property
    def sf_over_f0sq_tls_1hz(self) -> float:
        """TLS fractional frequency-noise PSD at 1 Hz offset from f0."""
        return self.sf_over_f0sq_tls_at_hz(1.0)

    @cached_property
    def s_deltaC_tls_1hz_F2_per_Hz(self) -> float:
        """Equivalent capacitance-fluctuation PSD at 1 Hz: S_dC = 4 C0^2 S_y."""
        c0 = self.C_res_F
        return 4.0 * (c0**2) * self.sf_over_f0sq_tls_1hz

    @cached_property
    def asd_deltaC_tls_1hz_F_per_rtHz(self) -> float:
        """Equivalent capacitance-fluctuation ASD at 1 Hz."""
        return sqrt(self.s_deltaC_tls_1hz_F2_per_Hz)

    @cached_property
    def sv_usb_tls_1hz_V2_per_Hz(self) -> float:
        """Equivalent USB capacitor-voltage PSD at f0 + 1 Hz from TLS C-noise."""
        c0 = self.C_res_F
        w0 = 2.0 * pi * self.f0_Hz
        return (self.I0_rms_A**2 / (w0 * w0 * c0 * c0)) * self.sf_over_f0sq_tls_1hz

    @cached_property
    def asd_v_usb_tls_1hz_V_per_rtHz(self) -> float:
        """Equivalent USB capacitor-voltage ASD at f0 + 1 Hz."""
        return sqrt(self.sv_usb_tls_1hz_V2_per_Hz)

    @cached_property
    def sv_usb_johnson_V2_per_Hz(self) -> float:
        """Johnson USB voltage PSD from Eq. (17) branch split."""
        return 0.5 * self.johnson_sv_V2_per_Hz

    @cached_property
    def asd_v_usb_johnson_V_per_rtHz(self) -> float:
        """Johnson USB voltage ASD."""
        return sqrt(self.sv_usb_johnson_V2_per_Hz)

    @cached_property
    def m_usb_tls_over_johnson_1hz(self) -> float:
        """USB voltage ASD ratio at 1 Hz offset: TLS / Johnson."""
        denom = self.asd_v_usb_johnson_V_per_rtHz
        if denom == 0.0:
            return float("nan")
        return self.asd_v_usb_tls_1hz_V_per_rtHz / denom

    def passive_tls_iq_transfer_phi_at_hz(self, nu_hz: float) -> complex:
        """Passive-resonator Q response to a unit phase-like IQ source.

        The reference uses the selected microwave resonator parameters
        (f0, Qr, Qi_eff, detuning), but removes electrothermal coupling,
        drive-power terms, island response, and feedback. This keeps the TLS
        source normalization independent of Kfb.
        """
        nu_eval_hz = 1.0 if nu_hz <= 0.0 else nu_hz
        w0 = 2.0 * pi * self.f0_Hz
        omega = 2.0 * pi * nu_eval_hz
        q = self.Qr
        qi = self.Qi_eff
        x = self.x

        m11 = (2.0j * omega * qi / w0) + (qi / q) + (4.0 * qi * q * x * x)
        m12 = -(4.0j * omega * qi * q * x / w0)
        m21 = +(4.0j * omega * qi * q * x / w0)
        m22 = (2.0j * omega * qi / w0) + (qi / q) + (4.0 * qi * q * x * x)
        y = np.linalg.solve(np.array(((m11, m12), (m21, m22)), dtype=complex), np.array((0.0 + 0.0j, 1.0j), dtype=complex))
        return complex(y[1])

    def tls_iq_source_asd_at_hz_per_rtHz(self, nu_hz: float) -> float:
        """IQ-source ASD that gives the requested TLS phase ASD passively."""
        nu_eval_hz = 1.0 if nu_hz <= 0.0 else nu_hz
        passive_gain = abs(self.passive_tls_iq_transfer_phi_at_hz(nu_eval_hz))
        if passive_gain == 0.0:
            return float("nan")
        return self.tls_phi_asd_at_hz_per_rtHz(nu_eval_hz) / passive_gain

    @cached_property
    def passive_tls_iq_transfer_phi_100hz_abs(self) -> float:
        """Passive Q-output gain from unit phase-like IQ source at 100 Hz."""
        return abs(self.passive_tls_iq_transfer_phi_at_hz(100.0))

    @cached_property
    def tls_iq_source_asd_100hz_per_rtHz(self) -> float:
        """TLS IQ source ASD calibrated to the passive 100 Hz phase ASD."""
        return self.tls_iq_source_asd_at_hz_per_rtHz(100.0)

    def n_tls_phi_at_hz(self, nu_hz: float) -> Tuple[complex, complex, complex, complex]:
        """TLS IQ source vector, normalized by a passive resonator reference."""
        nu_eval_hz = 1.0 if nu_hz <= 0.0 else nu_hz
        return (0.0 + 0.0j, 1j * self.tls_iq_source_asd_at_hz_per_rtHz(nu_eval_hz), 0.0 + 0.0j, 0.0 + 0.0j)

    def n_electronic_A_1(self) -> Tuple[complex, complex, complex, complex]:
        a1, a2, a3, a4 = self.n_johnson_A_1()
        me = self.me_electronic
        return (me * a1, me * a2, me * a3, me * a4)

    def n_electronic_A_2(self) -> Tuple[complex, complex, complex, complex]:
        a1, a2, a3, a4 = self.n_johnson_A_2()
        me = self.me_electronic
        return (me * a1, me * a2, me * a3, me * a4)

    def n_electronic_A(self) -> Tuple[complex, complex, complex, complex]:
        a = self.n_electronic_A_1()
        b = self.n_electronic_A_2()
        return tuple(ai + bi for ai, bi in zip(a, b))

    def n_electronic_phi_1(self) -> Tuple[complex, complex, complex, complex]:
        p1, p2, p3, p4 = self.n_johnson_phi_1()
        me = self.me_electronic
        return (me * p1, me * p2, me * p3, me * p4)

    def n_electronic_phi_2(self) -> Tuple[complex, complex, complex, complex]:
        p1, p2, p3, p4 = self.n_johnson_phi_2()
        me = self.me_electronic
        return (me * p1, me * p2, me * p3, me * p4)

    def n_electronic_phi(self) -> Tuple[complex, complex, complex, complex]:
        a = self.n_electronic_phi_1()
        b = self.n_electronic_phi_2()
        return tuple(ai + bi for ai, bi in zip(a, b))

    def m_matrix(
        self, f_hz: float = 1.0
    ) -> Tuple[Tuple[complex, complex, complex, complex], Tuple[complex, complex, complex, complex], Tuple[complex, complex, complex, complex], Tuple[complex, complex, complex, complex]]:
        """Compute 4x4 complex dual-KID M(omega) for states [r, phi, T1, T2]."""
        w0 = 2.0 * pi * self.f0_Hz
        omega = 2.0 * pi * f_hz
        q = self.Qr
        qi = self.Qi_eff
        x = self.x
        c1 = self.C_J_per_K
        c2 = max(self.heat_capacity2_eV_per_mK, 0.0) * 1.0e3 * J_PER_EV
        g1 = self.G_W_per_K
        g2 = max(self.G2_W_per_K, 0.0)
        t02 = self.T02_K if self.T02_K > 0.0 else self.T0_K
        r1 = max(self.R1_series_Ohm, 0.0)
        r2 = max(self.series_R2_Ohm, 0.0)
        rsum = r1 + r2
        p1 = self.P0_W if rsum <= 0.0 else self.P0_W * (r1 / rsum)
        p2 = 0.0 if rsum <= 0.0 else self.P0_W * (r2 / rsum)

        m11 = (2.0j * omega * qi / w0) + (qi / q) + self.beta_A + (4.0 * qi * q * x * x)
        m12 = -(4.0j * omega * qi * q * x / w0)
        m13 = -(2.0 * q * x * self.beta_A) + (2.0 * self.alpha_A / self.T0_K)
        m14 = 2.0 * self.alpha_A2 / t02

        m21 = +(4.0j * omega * qi * q * x / w0) - self.beta_phi
        m22 = (2.0j * omega * qi / w0) + (qi / q) + (4.0 * qi * q * x * x) + (2.0 * q * x * self.beta_phi)
        m23 = -(2.0 * self.alpha_phi / self.T0_K)
        m24 = -(2.0 * self.alpha_phi2 / t02)

        second_active = any(
            v > 0.0
            for v in (c2, g2, p2, abs(self.alpha_A2), abs(self.alpha_phi2), abs(self.feedback_heater_gain_W_per_rad))
        )

        m31 = -((1.0 + self.beta_A / 2.0) * p1)
        m32 = +(q * x * (self.beta_A + 2.0) * p1)
        m33 = (1.0j * omega * c1) + g1 - (p1 * self.alpha_A / self.T0_K)
        m34 = 0.0 + 0.0j

        if second_active:
            m41 = -((1.0 + self.beta_A / 2.0) * p2)
            m42 = +(q * x * (self.beta_A + 2.0) * p2) - self.feedback_heater_gain_W_per_rad
            m43 = 0.0 + 0.0j
            m44 = (1.0j * omega * c2) + g2 - (p2 * self.alpha_A2 / t02)
        else:
            m41 = 0.0 + 0.0j
            m42 = 0.0 + 0.0j
            m43 = 0.0 + 0.0j
            m44 = 1.0 + 0.0j

        return (
            (m11, m12, m13, m14),
            (m21, m22, m23, m24),
            (m31, m32, m33, m34),
            (m41, m42, m43, m44),
        )

    def m_matrix_array(self, f_hz: float = 1.0) -> np.ndarray:
        """M matrix as a 4x4 complex ndarray."""
        return np.array(self.m_matrix(f_hz), dtype=complex)

    @cached_property
    def second_kid_active(self) -> bool:
        c2 = max(self.heat_capacity2_eV_per_mK, 0.0) * 1.0e3 * J_PER_EV
        return any(
            v > 0.0
            for v in (
                c2,
                max(self.G2_W_per_K, 0.0),
                max(self.series_L2_H, 0.0),
                max(self.series_R2_Ohm, 0.0),
                abs(self.alpha_A2),
                abs(self.alpha_phi2),
                abs(self.feedback_heater_gain_W_per_rad),
            )
        )

    @cached_property
    def d0_matrix(self) -> np.ndarray:
        """D0 = M(0)."""
        d0 = self.m_matrix_array(0.0)
        if not self.second_kid_active:
            return d0[:3, :3]
        return d0

    @cached_property
    def d1_matrix(self) -> np.ndarray:
        """D1 = -i dM/domega, using central finite difference around omega=0.

        The derivative is with respect to angular frequency omega (rad/s).
        """
        df_hz = 1.0
        dw = 2.0 * pi * df_hz
        mp = self.m_matrix_array(+df_hz)
        mm = self.m_matrix_array(-df_hz)
        dmdw = (mp - mm) / (2.0 * dw)
        d1 = -1j * dmdw
        if not self.second_kid_active:
            return d1[:3, :3]
        return d1

    @cached_property
    def mt_matrix(self) -> np.ndarray:
        """Time-domain dynamics matrix Mt = -D1^{-1} D0."""
        return -np.linalg.inv(self.d1_matrix) @ self.d0_matrix

    @cached_property
    def mt_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of Mt (1/s)."""
        return np.linalg.eigvals(self.mt_matrix)

    @cached_property
    def mt_eigenvalues_sorted(self) -> np.ndarray:
        """Eigenvalues sorted by real part descending (worst stability first)."""
        vals = np.array(self.mt_eigenvalues, dtype=complex)
        idx = np.argsort(np.real(vals))[::-1]
        return vals[idx]

    @cached_property
    def mt_max_real_part(self) -> float:
        return float(np.max(np.real(self.mt_eigenvalues)))

    @cached_property
    def mt_stable(self) -> bool:
        """Stability criterion: all eigenvalue real parts must be negative."""
        return bool(np.all(np.real(self.mt_eigenvalues) < 0.0))

    @cached_property
    def mt_pulse_shortening_ratio(self) -> float:
        """Pulse shortening ratio: (G/C) normalized by slowest |lambda|.

        Defined only for stable eigenvalues. Returns NaN otherwise.
        """
        if not self.mt_stable:
            return float("nan")
        slowest = float(np.min(np.abs(self.mt_eigenvalues)))
        gc_freq = self.G_W_per_K / self.C_J_per_K
        if slowest == 0.0:
            return float("nan")
        return gc_freq / slowest

    @cached_property
    def asd_phi_phonon_full_per_rtHz(self) -> float:
        """Full matrix phonon phase ASD at f_demod."""
        return float(sqrt(abs(self.y_phonon_1[1]) ** 2 + abs(self.y_phonon_2[1]) ** 2))

    @cached_property
    def asd_phi_tls_100hz_model_per_rtHz(self) -> float:
        """TLS phase ASD at 100 Hz offset from modeled Sf/f0^2 and phase slope."""
        sf_f2_tls_100hz = self.sf_over_f0sq_tls_at_hz(100.0)
        return abs(self.f0_Hz * self.dphi_df_detuning_per_hz) * sqrt(sf_f2_tls_100hz)

    @cached_property
    def m_phonon_over_tls_phi(self) -> float:
        """Phase-ASD ratio: low-f phonon simple ASD / TLS ASD at 100 Hz offset."""
        denom = self.asd_phi_tls_100hz_model_per_rtHz
        if denom == 0.0:
            return float("nan")
        return self.asd_phi_phonon_simple_per_rtHz / denom

    @cached_property
    def core_rule3_ok(self) -> bool:
        """Rule 3: low-frequency phonon phase ASD should exceed TLS ASD at 100 Hz."""
        return bool(self.asd_phi_phonon_simple_per_rtHz > self.asd_phi_tls_100hz_model_per_rtHz)

    @cached_property
    def p_bifurcation_W(self) -> float:
        """Estimated bifurcation readout power.

        Uses common kinetic-inductance resonator scaling:
        P_bif ~ Qc * omega0 * E_star / (2 * Qr^3),
        where E_star is a device nonlinearity energy scale (input).
        """
        if self.Qr <= 0.0:
            return float("nan")
        w0 = 2.0 * pi * self.f0_Hz
        return (self.Qc * w0 * self.bifurcation_energy_scale_J) / (2.0 * (self.Qr**3))

    @cached_property
    def P0_W(self) -> float:
        """Resonator-dissipated power from generator drive via Eq. (3)-style mapping."""
        detuning_factor = 1.0 / (1.0 + 4.0 * (self.Qr**2) * (self.x**2))
        coupling_factor = (4.0 * self.Qc * self.Qi_eff) / ((self.Qc + self.Qi_eff) ** 2)
        return 0.5 * detuning_factor * coupling_factor * self.Pg_W

    @cached_property
    def P0_undetuned_W(self) -> float:
        """Worst-case on-resonance dissipated power for bifurcation checks."""
        coupling_factor = (4.0 * self.Qc * self.Qi_eff) / ((self.Qc + self.Qi_eff) ** 2)
        return 0.5 * coupling_factor * self.Pg_W

    @cached_property
    def Pg_W(self) -> float:
        """Generator/readout drive power from input drive level in dBm."""
        return 1.0e-3 * (10.0 ** (self.pg_drive_dBm / 10.0))

    @cached_property
    def p0_over_pbif_target(self) -> float:
        """Worst-case on-resonance dissipated-power fraction: P0(x=0) / P_bif."""
        if self.p_bifurcation_W <= 0.0:
            return float("nan")
        return self.P0_undetuned_W / self.p_bifurcation_W

    @cached_property
    def pg_to_p0_factor(self) -> float:
        """Transfer factor from generator power to dissipated resonator power."""
        if self.Pg_W == 0.0:
            return float("nan")
        return self.P0_W / self.Pg_W

    @cached_property
    def bifurcation_power_ratio(self) -> float:
        """Worst-case bifurcation margin using undetuned dissipated power."""
        if self.p_bifurcation_W <= 0.0:
            return float("nan")
        return self.P0_undetuned_W / self.p_bifurcation_W

    @cached_property
    def p_bifurcation_dBm(self) -> float:
        """Estimated bifurcation power in dBm."""
        if self.p_bifurcation_W <= 0.0:
            return float("nan")
        return 10.0 * log10(self.p_bifurcation_W / 1.0e-3)

    @cached_property
    def core_rule4_ok(self) -> bool:
        """Rule 4: stay below estimated bifurcation using worst-case detuning."""
        return bool(self.P0_undetuned_W < self.p_bifurcation_W)

    @cached_property
    def core_rule5_ok(self) -> bool:
        """Rule 5: operate at least halfway to bifurcation (soft floor)."""
        return bool(self.bifurcation_power_ratio > 0.5)

    @cached_property
    def core_rule6_ok(self) -> bool:
        """Rule 6: Pbif should fall in a typical 100 mK, ~1 GHz MKID dBm window."""
        return bool(self.pbif_typical_min_dBm <= self.p_bifurcation_dBm <= self.pbif_typical_max_dBm)

    @cached_property
    def core_rule7_ok(self) -> bool:
        """Rule 7: event temperature rise should exceed 1 mK."""
        return bool(self.deltaT_event_full_absorption_K > 1.0e-3)

    @cached_property
    def core_rule8_ok(self) -> bool:
        """Rule 8: event temperature rise should stay below 100 mK."""
        return bool(self.deltaT_event_full_absorption_K < 1.0e-1)

    @cached_property
    def core_rule9_ok(self) -> bool:
        """Rule 9: thermal fluctuation energy scale should be below target."""
        return bool(self.thermal_energy_fluct_rms_eV < self.thermal_energy_resolution_target_eV)

    @cached_property
    def kid2_thermal_headroom_K(self) -> float:
        """KID2 temperature elevation available for negative compensation."""
        if not self.second_kid_active:
            return float("nan")
        return self.T02_eff_K - self.Tb_K

    @cached_property
    def kid2_thermal_headroom_over_event_ratio(self) -> float:
        if not self.second_kid_active or self.deltaT_event_full_absorption_K <= 0:
            return float("nan")
        return self.kid2_thermal_headroom_K / self.deltaT_event_full_absorption_K

    @cached_property
    def dL1_dT_H_per_K(self) -> float:
        """KID1 inductance slope scale used for compensation headroom checks."""
        if self.T0_K <= 0.0:
            return float("nan")
        return self.alpha_phi * self.L_total_H / self.T0_K

    @cached_property
    def dL2_dT_H_per_K(self) -> float:
        """KID2 inductance slope scale used for compensation headroom checks."""
        if not self.second_kid_active or self.T02_eff_K <= 0.0:
            return float("nan")
        return self.alpha_phi2 * max(self.series_L2_H, 0.0) / self.T02_eff_K

    @cached_property
    def deltaL1_event_H(self) -> float:
        return self.dL1_dT_H_per_K * self.deltaT_event_full_absorption_K

    @cached_property
    def deltaL2_compensation_headroom_H(self) -> float:
        if not self.second_kid_active:
            return float("nan")
        return self.dL2_dT_H_per_K * self.kid2_thermal_headroom_K

    @cached_property
    def kid2_inductance_headroom_over_event_ratio(self) -> float:
        if not self.second_kid_active or self.deltaL1_event_H <= 0.0:
            return float("nan")
        return self.deltaL2_compensation_headroom_H / self.deltaL1_event_H

    @cached_property
    def core_rule10_ok(self) -> bool:
        """Rule 10: KID2 must have enough inductance headroom to cancel one event."""
        if not self.second_kid_active:
            return True
        return bool(self.deltaL2_compensation_headroom_H >= self.deltaL1_event_H)

    @cached_property
    def core_rule12_ok(self) -> bool:
        """Rule 12: Mt eigenvalues must all have negative real part."""
        return bool(self.mt_stable)

    @cached_property
    def core_rule13_ok(self) -> bool:
        """Rule 13: pileup probability should be below 0.5."""
        return bool(self.pileup_probability_max < 0.5)

    @cached_property
    def core_rule14_ok(self) -> bool:
        """Rule 14: NEP at 0 Hz should be within 1% of phonon NEP."""
        ratio = self.nep_phi_0hz_over_phonon_ratio
        if not np.isfinite(ratio):
            return False
        return bool(abs(ratio - 1.0) <= 0.01)

    @cached_property
    def core_rule15_ok(self) -> bool:
        """Rule 15: undetuned dissipated power must stay below bifurcation."""
        if not np.isfinite(self.p0_over_pbif_target):
            return False
        return bool(self.p0_over_pbif_target < 1.0)

    def estimates(self) -> Dict[str, float]:
        return {
            "f0_Hz": self.f0_Hz,
            "detuning_widths": self.detuning_widths,
            "detuning_Hz": self.detuning_Hz,
            "x": self.x,
            "f_demod_Hz": self.f_demod_Hz,
            "nep_sufficiency_percent": self.nep_sufficiency_percent,
            "count_rate_Hz": self.count_rate_Hz,
            "pileup_probability_max": self.pileup_probability_max,
            "nep_sufficient_frequency_hz": self.nep_sufficient_frequency_hz,
            "nep_sufficient_time_s": self.nep_sufficient_time_s,
            "ho_in_au_atomic_fraction": self.ho_in_au_atomic_fraction,
            "au_number_density_per_m3": self.au_number_density_per_m3,
            "ho_number_density_per_m3": self.ho_number_density_per_m3,
            "ho_decay_constant_per_s": self.ho_decay_constant_per_s,
            "ho_activity_per_m3_Hz": self.ho_activity_per_m3_Hz,
            "ho_decay_energy_J": self.ho_decay_energy_J,
            "absorber_volume_m3": self.absorber_volume_m3,
            "absorber_edge_m": self.absorber_edge_m,
            "absorber_length_m": self.absorber_length_m,
            "absorber_width_m": self.absorber_width_m,
            "absorber_thickness_m": self.absorber_thickness_m,
            "membrane_length_m": self.membrane_length_m,
            "membrane_width_m": self.membrane_width_m,
            "membrane_span_m": self.membrane_span_m,
            "cap_thickness_m": self.cap_thickness_m,
            "membrane_thickness_m": self.membrane_thickness_m,
            "leg_length_m": self.leg_length_m,
            "leg_thickness_m": self.leg_thickness_m,
            "leg_width_m": self.leg_width_m,
            "C_J_per_K": self.C_J_per_K,
            "C_eV_per_mK": self.C_eV_per_mK,
            "C_ho_eV_per_mK": self.C_ho_eV_per_mK,
            "G_W_per_K": self.G_W_per_K,
            "deltaT_abs_over_bath_setpoint_K": self.deltaT_abs_over_bath_setpoint_K,
            "deltaT_abs_over_bath_K": self.deltaT_abs_over_bath_K,
            "tbath_from_link_K": self.tbath_from_link_K,
            "deltaT_event_full_absorption_K": self.deltaT_event_full_absorption_K,
            "kid2_thermal_headroom_K": self.kid2_thermal_headroom_K,
            "kid2_thermal_headroom_over_event_ratio": self.kid2_thermal_headroom_over_event_ratio,
            "dL1_dT_H_per_K": self.dL1_dT_H_per_K,
            "dL2_dT_H_per_K": self.dL2_dT_H_per_K,
            "deltaL1_event_H": self.deltaL1_event_H,
            "deltaL2_compensation_headroom_H": self.deltaL2_compensation_headroom_H,
            "kid2_inductance_headroom_over_event_ratio": self.kid2_inductance_headroom_over_event_ratio,
            "dfr_dT_Hz_per_K": self.dfr_dT_Hz_per_K,
            "dT_dE_K_per_J": self.dT_dE_K_per_J,
            "dphi_dE_rad_per_J": self.dphi_dE_rad_per_J,
            "deltafr_event_Hz": self.deltafr_event_Hz,
            "deltaphi_event_rad": self.deltaphi_event_rad,
            "phonon_power_asd_device_W_per_rtHz": self.phonon_power_asd_device_W_per_rtHz,
            "phonon_temp_asd_device_K_per_rtHz": self.phonon_temp_asd_device_K_per_rtHz,
            "phonon_energy_asd_device_J_per_rtHz": self.phonon_energy_asd_device_J_per_rtHz,
            "asd_phi_phonon_simple_per_rtHz": self.asd_phi_phonon_simple_per_rtHz,
            "thermal_energy_fluct_rms_J": self.thermal_energy_fluct_rms_J,
            "thermal_energy_fluct_rms_eV": self.thermal_energy_fluct_rms_eV,
            "ho_decay_energy_eV": self.ho_decay_energy_eV,
            "tau_th_s": self.tau_th_s,
            "tau_target_from_rate_s": self.tau_target_from_rate_s,
            "tau_error_fraction": self.tau_error_fraction,
            "tau_res_s": self.tau_res_s,
            "tau_ratio_res_over_th": self.tau_ratio_res_over_th,
            "core_rule1_left_ratio": self.core_rule1_left_ratio,
            "core_rule1_right_ratio": self.core_rule1_right_ratio,
            "core_rule1_ok": float(self.core_rule1_ok),
            "core_rule2_ratio": self.core_rule2_ratio,
            "core_rule2_ok": float(self.core_rule2_ok),
            "core_rule3_ok": float(self.core_rule3_ok),
            "core_rule4_ok": float(self.core_rule4_ok),
            "core_rule5_ok": float(self.core_rule5_ok),
            "core_rule6_ok": float(self.core_rule6_ok),
            "core_rule7_ok": float(self.core_rule7_ok),
            "core_rule8_ok": float(self.core_rule8_ok),
            "core_rule9_ok": float(self.core_rule9_ok),
            "core_rule10_ok": float(self.core_rule10_ok),
            "core_rule12_ok": float(self.core_rule12_ok),
            "core_rule13_ok": float(self.core_rule13_ok),
            "core_rule14_ok": float(self.core_rule14_ok),
            "core_rule15_ok": float(self.core_rule15_ok),
            "L_geo_H": self.L_geo_H,
            "L_total_H": self.L_total_H,
            "C_res_F": self.C_res_F,
            "Z0_res_Ohm": self.Z0_res_Ohm,
            "R0_Ohm": self.R0_Ohm,
            "Qc": self.Qc,
            "Qr": self.Qr,
            "p_bifurcation_W": self.p_bifurcation_W,
            "p_bifurcation_dBm": self.p_bifurcation_dBm,
            "pg_drive_dBm": self.pg_drive_dBm,
            "p0_over_pbif_target": self.p0_over_pbif_target,
            "bifurcation_power_ratio": self.bifurcation_power_ratio,
            "tls_phi_asd_100hz_per_rtHz": self.tls_phi_asd_100hz_per_rtHz,
            "tls_beta": self.tls_beta,
            "passive_tls_iq_transfer_phi_100hz_abs": self.passive_tls_iq_transfer_phi_100hz_abs,
            "tls_iq_source_asd_100hz_per_rtHz": self.tls_iq_source_asd_100hz_per_rtHz,
            "sphi_johnson_full_per_hz": self.sphi_johnson_full_per_hz,
            "asd_phi_johnson_full_per_rtHz": self.asd_phi_johnson_full_per_rtHz,
            "sphi_tls_per_hz": self.sphi_tls_per_hz,
            "asd_phi_tls_per_rtHz": self.asd_phi_tls_per_rtHz,
            "sphi_electronic_per_hz": self.sphi_electronic_per_hz,
            "asd_phi_electronic_per_rtHz": self.asd_phi_electronic_per_rtHz,
            "sphi_total_per_hz": self.sphi_total_per_hz,
            "asd_phi_total_per_rtHz": self.asd_phi_total_per_rtHz,
            "asd_phi_tls_100hz_model_per_rtHz": self.asd_phi_tls_100hz_model_per_rtHz,
            "asd_phi_phonon_full_per_rtHz": self.asd_phi_phonon_full_per_rtHz,
            "phase_responsivity_rad_per_W": self.phase_responsivity_rad_per_W,
            "nep_phi_johnson_W_per_rtHz": self.nep_phi_johnson_W_per_rtHz,
            "nep_phi_tls_W_per_rtHz": self.nep_phi_tls_W_per_rtHz,
            "nep_phi_phonon_W_per_rtHz": self.nep_phi_phonon_W_per_rtHz,
            "nep_phi_electronic_W_per_rtHz": self.nep_phi_electronic_W_per_rtHz,
            "nep_phi_total_W_per_rtHz": self.nep_phi_total_W_per_rtHz,
            "nep_phi_phonon_0hz_W_per_rtHz": self.nep_phi_phonon_0hz_W_per_rtHz,
            "nep_phi_total_0hz_W_per_rtHz": self.nep_phi_total_0hz_W_per_rtHz,
            "nep_phi_0hz_over_phonon_ratio": self.nep_phi_0hz_over_phonon_ratio,
            "dphi_df_detuning_per_hz": self.dphi_df_detuning_per_hz,
            "sf_over_f0sq_johnson_full": self.sf_over_f0sq_johnson_full,
            "sf_over_f0sq_johnson_simple": self.sf_over_f0sq_johnson_simple,
            "m_phonon_over_johnson_phi": self.m_phonon_over_johnson_phi,
            "m_phonon_over_tls_phi": self.m_phonon_over_tls_phi,
            "sf_over_f0sq_tls_model": self.sf_over_f0sq_tls_model,
            "sf_over_f0sq_tls_1hz": self.sf_over_f0sq_tls_1hz,
            "I0_rms_A": self.I0_rms_A,
            "s_deltaC_tls_1hz_F2_per_Hz": self.s_deltaC_tls_1hz_F2_per_Hz,
            "asd_deltaC_tls_1hz_F_per_rtHz": self.asd_deltaC_tls_1hz_F_per_rtHz,
            "sv_usb_tls_1hz_V2_per_Hz": self.sv_usb_tls_1hz_V2_per_Hz,
            "asd_v_usb_tls_1hz_V_per_rtHz": self.asd_v_usb_tls_1hz_V_per_rtHz,
            "sv_usb_johnson_V2_per_Hz": self.sv_usb_johnson_V2_per_Hz,
            "asd_v_usb_johnson_V_per_rtHz": self.asd_v_usb_johnson_V_per_rtHz,
            "m_usb_tls_over_johnson_1hz": self.m_usb_tls_over_johnson_1hz,
            "P1_W": self.P1_W,
            "P2_W": self.P2_W,
            "heater2_dc_power_W": self.heater2_dc_power_W,
            "heater2_dc_power_dBm": self.heater2_dc_power_dBm,
            "alpha_A": self.alpha_A,
            "alpha_phi": self.alpha_phi,
            "alpha_A2": self.alpha_A2,
            "alpha_phi2": self.alpha_phi2,
            "beta_A": self.beta_A,
            "beta_phi": self.beta_phi,
            "Tc_K": self.Tc_K,
            "P0_W": self.P0_W,
            "Pg_W": self.Pg_W,
            "P0_undetuned_W": self.P0_undetuned_W,
            "pg_to_p0_factor": self.pg_to_p0_factor,
            "delta_J": self.delta_J,
            "eqp_J": self.eqp_J,
            "phonon_power_rms_W": self.phonon_power_rms_W,
            "johnson_voltage_rms_V": self.johnson_voltage_rms_V,
            "johnson_sv_V2_per_Hz": self.johnson_sv_V2_per_Hz,
            "M_e": self.me_electronic,
            "N_J_scale": self.nj_scale,
            "N_J_thermal_scale": self.nj_thermal_scale,
            "mt_eig1_real_per_s": float(np.real(self.mt_eigenvalues_sorted[0])),
            "mt_eig1_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[0])),
            "mt_eig2_real_per_s": float(np.real(self.mt_eigenvalues_sorted[1])),
            "mt_eig2_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[1])),
            "mt_eig3_real_per_s": float(np.real(self.mt_eigenvalues_sorted[2])),
            "mt_eig3_imag_per_s": float(np.imag(self.mt_eigenvalues_sorted[2])),
            "mt_max_real_part_per_s": self.mt_max_real_part,
            "mt_stable": float(self.mt_stable),
            "mt_pulse_shortening_ratio": self.mt_pulse_shortening_ratio,
        }


def version_1_inputs() -> SensorInputs:
    """Primary project input set."""
    return Version1SensorInputs()


def version_1_sensor() -> Sensor:
    """Sensor built from the 3x-count-rate preset."""
    return Sensor(inputs=version_1_inputs())

