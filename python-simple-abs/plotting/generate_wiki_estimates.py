"""Generate code-derived estimates for the wiki.

Outputs:
- python/outputs/wiki_estimates.json
- wiki/python-estimates.html
"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensor import version_1_sensor
OUT_JSON = ROOT / "outputs" / "wiki_estimates.json"
OUT_HTML = ROOT / "wiki" / "python-estimates.html"
OUT_DESIGN_HTML = ROOT / "wiki" / "design.html"
OUT_HTML_TOP = ROOT.parent / "wiki" / "python-estimates.html"
OUT_DESIGN_HTML_TOP = ROOT.parent / "wiki" / "design.html"


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.6g}"
    return f"{x:.6e}"


def _fmt_complex(z: complex) -> str:
    r = _fmt(float(z.real))
    i = _fmt(abs(float(z.imag)))
    sign = "+" if z.imag >= 0 else "-"
    return f"{r} {sign} {i}i"


def _pf_html(x: float) -> str:
    return '<span class="pass">Pass</span>' if float(x) >= 0.5 else '<span class="fail">Fail</span>'


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML_TOP.parent.mkdir(parents=True, exist_ok=True)
    s = version_1_sensor()
    est = s.estimates()
    input_keys = [f.name for f in fields(s.inputs)]

    units = {
        "T0_K": "K",
        "Tb_K": "K",
        "count_rate_Hz": "Hz",
        "deltaT_abs_over_bath_setpoint_K": "K",
        "pileup_probability_max": "1",
        "ho_in_au_atomic_fraction": "1",
        "ho_activity_per_m3_Hz": "Hz/m^3",
        "au_number_density_per_m3": "1/m^3",
        "ho_number_density_per_m3": "1/m^3",
        "ho_decay_constant_per_s": "1/s",
        "ho_decay_energy_J": "J",
        "ho_decay_energy_eV": "eV",
        "absorber_length_m": "m",
        "absorber_width_m": "m",
        "absorber_thickness_m": "m",
        "absorber_island_length_m": "m",
        "absorber_island_width_m": "m",
        "absorber_island_area_m2": "m^2",
        "membrane_volume_m3": "m^3",
        "membrane_heat_capacity_J_per_K": "J/K",
        "membrane_heat_capacity_eV_per_mK": "eV/mK",
        "absorber_edge_m": "m",
        "kid_length_m": "m",
        "kid_width_m": "m",
        "membrane_margin_m": "m",
        "leg_count": "count",
        "leg_width_m": "m",
        "leg_thickness_m": "m",
        "cap_thickness_m": "m",
        "membrane_thickness_m": "m",
        "cv_absorber_J_per_m3K": "J/(m^3 K)",
        "kappa_leg_W_per_mK": "W/(m K)",
        "thermal_link_exponent_n": "1",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": "1/Hz^(1/2)",
        "tls_phi_asd_100hz_per_rtHz": "1/Hz^(1/2)",
        "tls_beta": "1",
        "passive_tls_iq_transfer_phi_100hz_abs": "1",
        "tls_iq_source_asd_100hz_per_rtHz": "1/Hz^(1/2)",
        "f0_Hz": "Hz",
        "Qr": "1",
        "Qi": "1",
        "tau_qp_s": "s",
        "kinetic_inductance_fraction": "1",
        "kid_trace_length_m": "m",
        "kid_trace_width_m": "m",
        "alpha_A": "1",
        "alpha_phi": "1",
        "beta_A": "1",
        "beta_phi": "1",
        "Tc_K": "K",
        "P0_W": "W",
        "P0_undetuned_W": "W",
        "Pg_W": "W",
        "pg_to_p0_factor": "1",
        "p0_over_pbif_target": "1",
        "bifurcation_energy_scale_J": "J",
        "pbif_typical_min_dBm": "dBm",
        "pbif_typical_max_dBm": "dBm",
        "thermal_energy_resolution_target_eV": "eV",
        "delta_J": "J",
        "eqp_J": "J",
        "detuning_widths": "fr/Qr",
        "detuning_pid_gain_Hz_per_rad": "Hz/rad",
        "detuning_pid_integrator_time_s": "s",
        "detuning_pid_derivative_time_s": "s",
        "detuning_pid_derivative_filter_factor": "1",
        "amplifier_noise_temperature_K": "K",
        "detuning_pid_transfer_Hz_per_rad": "Hz/rad",
        "detuning_pid_transfer_dx_per_rad": "1/rad",
        "dx_response_to_event_power_at_f_demod": "1/W",
        "nep_sufficiency_percent": "%",
        "detuning_Hz": "Hz",
        "x": "1",
        "f_demod_Hz": "Hz",
        "T02_K": "K",
        "heat_capacity2_eV_per_mK": "eV/mK",
        "G2_W_per_K": "W/K",
        "alpha_A2": "1",
        "alpha_phi2": "1",
        "series_L2_H": "H",
        "series_R2_Ohm": "Ohm",
        "heater2_offset_dBm": "dBm",
        "feedback_heater_gain_W_per_rad": "W/rad",
        "absorber_volume_m3": "m^3",
        "membrane_length_m": "m",
        "membrane_width_m": "m",
        "membrane_span_m": "m",
        "leg_length_m": "m",
        "C_J_per_K": "J/K",
        "C_eV_per_mK": "eV/mK",
        "C_ho_eV_per_mK": "eV/mK",
        "G_W_per_K": "W/K",
        "deltaT_abs_over_bath_K": "K",
        "tbath_from_link_K": "K",
        "deltaT_event_full_absorption_K": "K",
        "event_peak_temperature_K": "K",
        "kid2_thermal_headroom_K": "K",
        "kid2_thermal_headroom_over_event_ratio": "1",
        "dL1_dT_H_per_K": "H/K",
        "dL2_dT_H_per_K": "H/K",
        "deltaL1_event_H": "H",
        "deltaL2_compensation_headroom_H": "H",
        "kid2_inductance_headroom_over_event_ratio": "1",
        "dfr_dT_Hz_per_K": "Hz/K",
        "dT_dE_K_per_J": "K/J",
        "dphi_dE_rad_per_J": "rad/J",
        "deltafr_event_Hz": "Hz",
        "deltaphi_event_rad": "rad",
        "phonon_power_asd_device_W_per_rtHz": "W/Hz^(1/2)",
        "phonon_temp_asd_device_K_per_rtHz": "K/Hz^(1/2)",
        "phonon_energy_asd_device_J_per_rtHz": "J/Hz^(1/2)",
        "asd_phi_phonon_simple_per_rtHz": "1/Hz^(1/2)",
        "thermal_energy_fluct_rms_J": "J",
        "thermal_energy_fluct_rms_eV": "eV",
        "tau_th_s": "s",
        "nep_sufficient_frequency_hz": "Hz",
        "nep_sufficient_time_s": "s",
        "tau_target_from_rate_s": "s",
        "tau_error_fraction": "1",
        "tau_res_s": "s",
        "tau_ratio_res_over_th": "1",
        "core_rule1_left_ratio": "1",
        "core_rule1_right_ratio": "1",
        "core_rule1_ok": "1",
        "core_rule2_ratio": "1",
        "core_rule2_ok": "1",
        "core_rule3_ok": "1",
        "core_rule4_ok": "1",
        "core_rule5_ok": "1",
        "core_rule6_ok": "1",
        "core_rule7_ok": "1",
        "core_rule8_ok": "1",
        "core_rule9_ok": "1",
        "core_rule10_ok": "1",
        "core_rule11_ok": "1",
        "core_rule12_ok": "1",
        "core_rule13_ok": "1",
        "core_rule14_ok": "1",
        "core_rule15_ok": "1",
        "L_geo_H": "H",
        "L_total_H": "H",
        "C_res_F": "F",
        "Z0_res_Ohm": "Ohm",
        "Z0_readout_Ohm": "Ohm",
        "readout_circle_radius_V": "V",
        "R0_Ohm": "Ohm",
        "Qc": "1",
        "p_bifurcation_W": "W",
        "p_bifurcation_dBm": "dBm",
        "bifurcation_power_ratio": "1",
        "sphi_johnson_full_per_hz": "1/Hz",
        "sphi_tls_per_hz": "1/Hz",
        "sphi_amplifier_per_hz": "1/Hz",
        "asd_phi_amplifier_per_rtHz": "1/Hz^(1/2)",
        "asd_phi_tls_per_rtHz": "1/Hz^(1/2)",
        "asd_phi_tls_100hz_model_per_rtHz": "1/Hz^(1/2)",
        "asd_phi_phonon_full_per_rtHz": "1/Hz^(1/2)",
        "dphi_df_detuning_per_hz": "rad/Hz",
        "sf_over_f0sq_johnson_full": "1/Hz",
        "sf_over_f0sq_johnson_simple": "1/Hz",
        "m_phonon_over_johnson_phi": "1",
        "m_phonon_over_tls_phi": "1",
        "sf_over_f0sq_tls_model": "1/Hz",
        "sf_over_f0sq_tls_1hz": "1/Hz",
        "I0_rms_A": "A",
        "s_deltaC_tls_1hz_F2_per_Hz": "F^2/Hz",
        "asd_deltaC_tls_1hz_F_per_rtHz": "F/Hz^(1/2)",
        "s_deltaC_over_C_tls_100hz_per_Hz": "1/Hz",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": "1/Hz^(1/2)",
        "sv_usb_tls_1hz_V2_per_Hz": "V^2/Hz",
        "asd_v_usb_tls_1hz_V_per_rtHz": "V/Hz^(1/2)",
        "sv_usb_johnson_V2_per_Hz": "V^2/Hz",
        "asd_v_usb_johnson_V_per_rtHz": "V/Hz^(1/2)",
        "m_usb_tls_over_johnson_1hz": "1",
        "P1_W": "W",
        "P2_W": "W",
        "heater2_dc_power_W": "W",
        "heater2_dc_power_dBm": "dBm",
        "phonon_power_rms_W": "W",
        "johnson_voltage_rms_V": "V",
        "johnson_sv_V2_per_Hz": "V^2/Hz",
        "amplifier_sideband_voltage_asd_V_per_rtHz": "V/Hz^(1/2)",
        "amplifier_normalized_asd_per_rtHz": "1/Hz^(1/2)",
        "nep_phi_amplifier_W_per_rtHz": "W/Hz^(1/2)",
        "M_e": "1",
        "N_J_scale": "s^(1/2)",
        "N_J_thermal_scale": "W s^(1/2)",
        "mt_eig1_per_s": "1/s",
        "mt_eig2_per_s": "1/s",
        "mt_eig3_per_s": "1/s",
        "mt_eig4_per_s": "1/s",
        "mt_max_real_part_per_s": "1/s",
        "mt_stable": "1",
        "mt_pulse_shortening_ratio": "1",
    }
    symbols = {
        "T0_K": r"\(T_0\)",
        "Tb_K": r"\(T_b\)",
        "count_rate_Hz": r"\(R\)",
        "deltaT_abs_over_bath_setpoint_K": r"\(\Delta T_{\mathrm{abs-bath,set}}\)",
        "pileup_probability_max": r"\(P_{\mathrm{pileup,reject}}\)",
        "ho_in_au_atomic_fraction": r"\(x_{\mathrm{Ho/Au}}\)",
        "ho_activity_per_m3_Hz": r"\(A_{\mathrm{Ho}}\)",
        "au_number_density_per_m3": r"\(n_{\mathrm{Au}}\)",
        "ho_number_density_per_m3": r"\(n_{\mathrm{Ho}}\)",
        "ho_decay_constant_per_s": r"\(\lambda_{\mathrm{Ho}}\)",
        "ho_decay_energy_J": r"\(E_{\mathrm{Ho}}\)",
        "ho_decay_energy_eV": r"\(E_{\mathrm{Ho,eV}}\)",
        "absorber_length_m": r"\(L_{\mathrm{abs}}\)",
        "absorber_width_m": r"\(W_{\mathrm{abs}}\)",
        "absorber_thickness_m": r"\(t_{\mathrm{abs}}\)",
        "absorber_island_length_m": r"\(L_{\mathrm{island}}\)",
        "absorber_island_width_m": r"\(W_{\mathrm{island}}\)",
        "absorber_island_area_m2": r"\(A_{\mathrm{island}}\)",
        "membrane_volume_m3": r"\(V_{\mathrm{mem}}\)",
        "membrane_heat_capacity_J_per_K": r"\(C_{\mathrm{mem}}(T_1)\)",
        "membrane_heat_capacity_eV_per_mK": r"\(C_{\mathrm{mem,eV/mK}}(T_1)\)",
        "absorber_edge_m": r"\(a_{\mathrm{abs}}\)",
        "kid_length_m": r"\(L_{\mathrm{KID}}\)",
        "kid_width_m": r"\(W_{\mathrm{KID}}\)",
        "membrane_margin_m": r"\(\Delta_{\mathrm{mem}}\)",
        "leg_count": r"\(N_{\mathrm{leg}}\)",
        "leg_width_m": r"\(w_{\mathrm{leg}}\)",
        "leg_thickness_m": r"\(t_{\mathrm{leg}}\)",
        "cap_thickness_m": r"\(t_{\mathrm{cap}}\)",
        "membrane_thickness_m": r"\(t_{\mathrm{mem}}\)",
        "cv_absorber_J_per_m3K": r"\(c_{V,\mathrm{abs}}\)",
        "kappa_leg_W_per_mK": r"\(\kappa_{\mathrm{leg}}\)",
        "thermal_link_exponent_n": r"\(n_{\mathrm{link}}\)",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": r"\(\sqrt{S_{\delta C/C,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "tls_phi_asd_100hz_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "tls_beta": r"\(\beta_{\mathrm{TLS}}\)",
        "passive_tls_iq_transfer_phi_100hz_abs": r"\(|H_{\mathrm{TLS,passive}}(100\,\mathrm{Hz})|\)",
        "tls_iq_source_asd_100hz_per_rtHz": r"\(\sqrt{S_{N,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "f0_Hz": r"\(f_0\)",
        "Qr": r"\(Q_r\)",
        "Qi": r"\(Q_i\)",
        "tau_qp_s": r"\(\tau_{qp}\)",
        "kinetic_inductance_fraction": r"\(\alpha_k\)",
        "kid_trace_length_m": r"\(\ell\)",
        "kid_trace_width_m": r"\(w\)",
        "alpha_A": r"\(\alpha_A\)",
        "alpha_phi": r"\(\alpha_{\phi}\)",
        "beta_A": r"\(\beta_A\)",
        "beta_phi": r"\(\beta_{\phi}\)",
        "Tc_K": r"\(T_c\)",
        "R0_Ohm": r"\(R_0\)",
        "P0_W": r"\(P_0\)",
        "P0_undetuned_W": r"\(P_{0,x=0}\)",
        "Pg_W": r"\(P_g\)",
        "pg_to_p0_factor": r"\(P_0/P_g\)",
        "p0_over_pbif_target": r"\(P_{0,x=0}/P_{\mathrm{bif}}\)",
        "bifurcation_energy_scale_J": r"\(E_*\)",
        "pbif_typical_min_dBm": r"\(P_{\mathrm{bif,min}}^{\mathrm{typ}}\)",
        "pbif_typical_max_dBm": r"\(P_{\mathrm{bif,max}}^{\mathrm{typ}}\)",
        "thermal_energy_resolution_target_eV": r"\(\sigma_{E,\mathrm{target}}\)",
        "delta_J": r"\(\Delta\)",
        "eqp_J": r"\(E_{qp}\)",
        "detuning_widths": r"\(xQ_r\)",
        "detuning_pid_gain_Hz_per_rad": r"\(K_f\)",
        "detuning_pid_integrator_time_s": r"\(\tau_i\)",
        "detuning_pid_derivative_time_s": r"\(\tau_d\)",
        "detuning_pid_derivative_filter_factor": r"\(N_d\)",
        "amplifier_noise_temperature_K": r"\(T_A\)",
        "detuning_pid_transfer_Hz_per_rad": r"\(|H_f|\)",
        "detuning_pid_transfer_dx_per_rad": r"\(|H_x|\)",
        "dx_response_to_event_power_at_f_demod": r"\(|dx/P|\)",
        "nep_sufficiency_percent": r"\(p_{\mathrm{NEP,suff}}\)",
        "detuning_Hz": r"\(\delta f\)",
        "x": r"\(x\)",
        "f_demod_Hz": r"\(f_{\mathrm{demod}}\)",
        "T02_K": r"\(T_{02}\)",
        "heat_capacity2_eV_per_mK": r"\(C_2\)",
        "G2_W_per_K": r"\(G_2\)",
        "alpha_A2": r"\(\alpha_{A2}\)",
        "alpha_phi2": r"\(\alpha_{\phi2}\)",
        "series_L2_H": r"\(L_2\)",
        "series_R2_Ohm": r"\(R_2\)",
        "heater2_offset_dBm": r"\(P_{\mathrm{heater2,off,dBm}}\)",
        "feedback_heater_gain_W_per_rad": r"\(K_{\mathrm{fb}}\)",
        "absorber_volume_m3": r"\(V_{\mathrm{abs}}\)",
        "membrane_length_m": r"\(L_{\mathrm{mem}}\)",
        "membrane_width_m": r"\(W_{\mathrm{mem}}\)",
        "membrane_span_m": r"\(S_{\mathrm{mem}}\)",
        "leg_length_m": r"\(L_{\mathrm{leg}}\)",
        "C_J_per_K": r"\(C\)",
        "C_eV_per_mK": r"\(C_{\mathrm{eV/mK}}\)",
        "C_ho_eV_per_mK": r"\(C_{\mathrm{Ho,eV/mK}}\)",
        "G_W_per_K": r"\(G\)",
        "deltaT_abs_over_bath_K": r"\(\Delta T_{\mathrm{abs-bath}}\)",
        "tbath_from_link_K": r"\(T_{\mathrm{bath}}\)",
        "deltaT_event_full_absorption_K": r"\(\Delta T_{\mathrm{event}}\)",
        "event_peak_temperature_K": r"\(T_{\mathrm{event,peak}}\)",
        "kid2_thermal_headroom_K": r"\(\Delta T_{2,\mathrm{head}}\)",
        "kid2_thermal_headroom_over_event_ratio": r"\(\Delta T_{2,\mathrm{head}}/\Delta T_{\mathrm{event}}\)",
        "dL1_dT_H_per_K": r"\(dL_1/dT_1\)",
        "dL2_dT_H_per_K": r"\(dL_2/dT_2\)",
        "deltaL1_event_H": r"\(\Delta L_{1,\mathrm{event}}\)",
        "deltaL2_compensation_headroom_H": r"\(\Delta L_{2,\mathrm{head}}\)",
        "kid2_inductance_headroom_over_event_ratio": r"\(\Delta L_{2,\mathrm{head}}/\Delta L_{1,\mathrm{event}}\)",
        "dfr_dT_Hz_per_K": r"\(df_r/dT\)",
        "dT_dE_K_per_J": r"\(dT/dE\)",
        "dphi_dE_rad_per_J": r"\(d\phi/dE\)",
        "deltafr_event_Hz": r"\(\Delta f_{r,\mathrm{event}}\)",
        "deltaphi_event_rad": r"\(\Delta\phi_{\mathrm{event}}\)",
        "phonon_power_asd_device_W_per_rtHz": r"\(\sqrt{S_{P,\mathrm{ph}}}(T_0)\)",
        "phonon_temp_asd_device_K_per_rtHz": r"\(\sqrt{S_{T,\mathrm{ph}}}(T_0)\)",
        "phonon_energy_asd_device_J_per_rtHz": r"\(\sqrt{S_{E,\mathrm{ph}}}(T_0)\)",
        "asd_phi_phonon_simple_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{simple}}(T_0)\)",
        "thermal_energy_fluct_rms_J": r"\(\sigma_{E,\mathrm{th}}\)",
        "thermal_energy_fluct_rms_eV": r"\(\sigma_{E,\mathrm{th,eV}}\)",
        "tau_th_s": r"\(\tau_{\mathrm{th}}\)",
        "nep_sufficient_frequency_hz": r"\(f_{\mathrm{NEP,suff}}\)",
        "nep_sufficient_time_s": r"\(\tau_{\mathrm{NEP,suff}}\)",
        "tau_target_from_rate_s": r"\(\tau_{\mathrm{target}}\)",
        "tau_error_fraction": r"\(\epsilon_{\tau}\)",
        "tau_res_s": r"\(\tau_{\mathrm{res}}\)",
        "tau_ratio_res_over_th": r"\(\tau_{\mathrm{res}}/\tau_{\mathrm{th}}\)",
        "core_rule1_left_ratio": r"\(\tau_{qp}/\tau_{\mathrm{res}}\)",
        "core_rule1_right_ratio": r"\(\tau_{\mathrm{res}}/\tau_{\mathrm{th}}\)",
        "core_rule1_ok": r"\(\mathbb{1}_{\mathrm{rule1}}\)",
        "core_rule2_ratio": r"\(\tau_{\mathrm{res}}/\tau_{\mathrm{th}}\)",
        "core_rule2_ok": r"\(\mathbb{1}_{\mathrm{rule2}}\)",
        "core_rule3_ok": r"\(\mathbb{1}_{\mathrm{rule3}}\)",
        "core_rule4_ok": r"\(\mathbb{1}_{\mathrm{rule4}}\)",
        "core_rule5_ok": r"\(\mathbb{1}_{\mathrm{rule5}}\)",
        "core_rule6_ok": r"\(\mathbb{1}_{\mathrm{rule6}}\)",
        "core_rule7_ok": r"\(\mathbb{1}_{\mathrm{rule7}}\)",
        "core_rule8_ok": r"\(\mathbb{1}_{\mathrm{rule8}}\)",
        "core_rule9_ok": r"\(\mathbb{1}_{\mathrm{rule9}}\)",
        "core_rule10_ok": r"\(\mathbb{1}_{\mathrm{rule10}}\)",
        "core_rule11_ok": r"\(\mathbb{1}_{\mathrm{rule11}}\)",
        "core_rule12_ok": r"\(\mathbb{1}_{\mathrm{rule12}}\)",
        "core_rule13_ok": r"\(\mathbb{1}_{\mathrm{rule13}}\)",
        "core_rule14_ok": r"\(\mathbb{1}_{\mathrm{rule14}}\)",
        "core_rule15_ok": r"\(\mathbb{1}_{\mathrm{rule15}}\)",
        "L_geo_H": r"\(L_g\)",
        "L_total_H": r"\(L_{\mathrm{tot}}\)",
        "C_res_F": r"\(C_{\mathrm{res}}\)",
        "Z0_res_Ohm": r"\(Z_0\)",
        "Z0_readout_Ohm": r"\(Z_{\mathrm{readout}}\)",
        "readout_circle_radius_V": r"\(r_V\)",
        "R0_Ohm": r"\(R_0\)",
        "Qc": r"\(Q_c\)",
        "p_bifurcation_W": r"\(P_{\mathrm{bif}}\)",
        "p_bifurcation_dBm": r"\(P_{\mathrm{bif,dBm}}\)",
        "bifurcation_power_ratio": r"\(P_{0,x=0}/P_{\mathrm{bif}}\)",
        "sphi_johnson_full_per_hz": r"\(S_{\phi,J}^{\mathrm{full}}\)",
        "sphi_tls_per_hz": r"\(S_{\phi,\mathrm{TLS}}\)",
        "sphi_amplifier_per_hz": r"\(S_{\phi,\mathrm{amp}}\)",
        "asd_phi_amplifier_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{amp}}}\)",
        "asd_phi_tls_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{TLS}}}\)",
        "asd_phi_tls_100hz_model_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "asd_phi_phonon_full_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{full}}\)",
        "dphi_df_detuning_per_hz": r"\(\left|d\phi/df\right|_{x}\)",
        "sf_over_f0sq_johnson_full": r"\(\left(S_f/f_0^2\right)_{J,\mathrm{full}}\)",
        "sf_over_f0sq_johnson_simple": r"\(\left(S_f/f_0^2\right)_{J,\mathrm{simple}}\)",
        "m_phonon_over_johnson_phi": r"\(M_{\mathrm{ph}/J,\phi}\)",
        "m_phonon_over_tls_phi": r"\(M_{\mathrm{ph}/\mathrm{TLS},\phi}\)",
        "sf_over_f0sq_tls_model": r"\(\left(S_f/f_0^2\right)_{\mathrm{TLS}}\)",
        "sf_over_f0sq_tls_1hz": r"\(\left(S_f/f_0^2\right)_{\mathrm{TLS},1\mathrm{Hz}}\)",
        "I0_rms_A": r"\(I_0\)",
        "s_deltaC_tls_1hz_F2_per_Hz": r"\(S_{\delta C,\mathrm{TLS}}(1\,\mathrm{Hz})\)",
        "asd_deltaC_tls_1hz_F_per_rtHz": r"\(\sqrt{S_{\delta C,\mathrm{TLS}}}(1\,\mathrm{Hz})\)",
        "s_deltaC_over_C_tls_100hz_per_Hz": r"\(S_{\delta C/C,\mathrm{TLS}}(100\,\mathrm{Hz})\)",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": r"\(\sqrt{S_{\delta C/C,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "sv_usb_tls_1hz_V2_per_Hz": r"\(S_{V,\mathrm{USB},\mathrm{TLS}}(1\,\mathrm{Hz})\)",
        "asd_v_usb_tls_1hz_V_per_rtHz": r"\(\sqrt{S_{V,\mathrm{USB},\mathrm{TLS}}}(1\,\mathrm{Hz})\)",
        "sv_usb_johnson_V2_per_Hz": r"\(S_{V,\mathrm{USB},J}\)",
        "asd_v_usb_johnson_V_per_rtHz": r"\(\sqrt{S_{V,\mathrm{USB},J}}\)",
        "m_usb_tls_over_johnson_1hz": r"\(M_{\mathrm{USB},\mathrm{TLS}/J}(1\,\mathrm{Hz})\)",
        "P1_W": r"\(P_1\)",
        "P2_W": r"\(P_2\)",
        "heater2_dc_power_W": r"\(P_{\mathrm{heater},2}\)",
        "heater2_dc_power_dBm": r"\(P_{\mathrm{heater},2,\mathrm{dBm}}\)",
        "phonon_power_rms_W": r"\(P_{\mathrm{ph,RMS}}\)",
        "johnson_voltage_rms_V": r"\(V_{e}\)",
        "johnson_sv_V2_per_Hz": r"\(S_{V,J}\)",
        "amplifier_sideband_voltage_asd_V_per_rtHz": r"\(V_A\)",
        "amplifier_normalized_asd_per_rtHz": r"\(\sigma_{\mathrm{amp}}\)",
        "nep_phi_amplifier_W_per_rtHz": r"\(\mathrm{NEP}_{\phi,\mathrm{amp}}\)",
        "M_e": r"\(M_e\)",
        "N_J_scale": r"\(N_{J,\mathrm{scale}}\)",
        "N_J_thermal_scale": r"\(N_{J,\mathrm{th}}\)",
        "mt_eig1_per_s": r"\(\lambda_1(M_t)\)",
        "mt_eig2_per_s": r"\(\lambda_2(M_t)\)",
        "mt_eig3_per_s": r"\(\lambda_3(M_t)\)",
        "mt_eig4_per_s": r"\(\lambda_4(M_t)\)",
        "mt_max_real_part_per_s": r"\(\max\Re[\lambda(M_t)]\)",
        "mt_stable": r"\(\mathbb{1}_{\mathrm{stable}}\)",
        "mt_pulse_shortening_ratio": r"\(\rho_{\mathrm{short}}\)",
    }
    formulas = {
        "au_number_density_per_m3": r"\(n_{\mathrm{Au}}=\dfrac{\rho_{\mathrm{Au}}}{M_{\mathrm{Au}}}N_A\)",
        "ho_number_density_per_m3": r"\(n_{\mathrm{Ho}}=x_{\mathrm{Ho/Au}}\,n_{\mathrm{Au}}\)",
        "ho_decay_constant_per_s": r"\(\lambda_{\mathrm{Ho}}=\dfrac{\ln 2}{T_{1/2}}\)",
        "ho_activity_per_m3_Hz": r"\(A_{\mathrm{Ho}}=\lambda_{\mathrm{Ho}}\,n_{\mathrm{Ho}}\)",
        "absorber_volume_m3": r"\(V_{\mathrm{abs}}=\dfrac{R}{A_{\mathrm{Ho}}}\)",
        "absorber_edge_m": r"\(a_{\mathrm{abs}}=V_{\mathrm{abs}}^{1/3}\)",
        "absorber_length_m": r"\(L_{\mathrm{abs}}=a_{\mathrm{abs}}\)",
        "absorber_width_m": r"\(W_{\mathrm{abs}}=a_{\mathrm{abs}}\)",
        "absorber_thickness_m": r"\(t_{\mathrm{abs}}=a_{\mathrm{abs}}\)",
        "absorber_island_length_m": r"\(L_{\mathrm{island}}=L_{\mathrm{mem}}\)",
        "absorber_island_width_m": r"\(W_{\mathrm{island}}=W_{\mathrm{mem}}\)",
        "absorber_island_area_m2": r"\(A_{\mathrm{island}}=L_{\mathrm{island}}W_{\mathrm{island}}\)",
        "membrane_volume_m3": r"\(V_{\mathrm{mem}}=L_{\mathrm{mem}}W_{\mathrm{mem}}t_{\mathrm{mem}}\)",
        "membrane_heat_capacity_J_per_K": r"\(C_{\mathrm{mem}}(T_1)=c_{V}(T_1)\,V_{\mathrm{mem}}\)",
        "membrane_heat_capacity_eV_per_mK": r"\(C_{\mathrm{mem,eV/mK}}(T_1)=\dfrac{C_{\mathrm{mem}}(T_1)}{q_e\cdot 10^3}\)",
        "membrane_length_m": r"\(L_{\mathrm{mem}}=L_{\mathrm{abs}}+L_{\mathrm{KID}}+\Delta_{\mathrm{mem}}\)",
        "membrane_width_m": r"\(W_{\mathrm{mem}}=W_{\mathrm{abs}}+W_{\mathrm{KID}}+\Delta_{\mathrm{mem}}\)",
        "membrane_span_m": r"\(S_{\mathrm{mem}}=\max(L_{\mathrm{mem}},W_{\mathrm{mem}})\)",
        "leg_length_m": r"\(L_{\mathrm{leg}}=\dfrac{N_{\mathrm{leg}}\kappa_{\mathrm{leg}}w_{\mathrm{leg}}t_{\mathrm{leg}}}{G}\)",
        "leg_thickness_m": r"\(t_{\mathrm{leg}}=t_{\mathrm{mem}}\)",
        "C_J_per_K": r"\(C=c_{V,\mathrm{abs}}V_{\mathrm{abs}}\)",
        "C_eV_per_mK": r"\(C_{\mathrm{eV/mK}}=\dfrac{C}{q_e\cdot 10^3}\)",
        "C_ho_eV_per_mK": r"\(C_{\mathrm{Ho,eV/mK}}=\dfrac{C_{\mathrm{Ho}}}{q_e\cdot 10^3}\)",
        "G_W_per_K": r"\(G=\dfrac{P_0}{\Delta T_{\mathrm{abs-bath,set}}}\)",
        "deltaT_abs_over_bath_K": r"\(\Delta T_{\mathrm{abs-bath}}=\dfrac{P_0}{G}\)",
        "tbath_from_link_K": r"\(T_{\mathrm{bath}}=T_0-\Delta T_{\mathrm{abs-bath}}\)",
        "deltaT_event_full_absorption_K": r"\(\Delta T_{\mathrm{event}}=\dfrac{E_{\mathrm{Ho}}}{C}\)",
        "kid2_thermal_headroom_K": r"\(\Delta T_{2,\mathrm{head}}=T_{02}-T_b\)",
        "kid2_thermal_headroom_over_event_ratio": r"\(\Delta T_{2,\mathrm{head}}/\Delta T_{\mathrm{event}}\)",
        "dL1_dT_H_per_K": r"\(\dfrac{dL_1}{dT_1}\propto\dfrac{\alpha_{\phi1}L_1}{T_1}\)",
        "dL2_dT_H_per_K": r"\(\dfrac{dL_2}{dT_2}\propto\dfrac{\alpha_{\phi2}L_2}{T_2}\)",
        "deltaL1_event_H": r"\(\Delta L_{1,\mathrm{event}}=\dfrac{dL_1}{dT_1}\Delta T_{\mathrm{event}}\)",
        "deltaL2_compensation_headroom_H": r"\(\Delta L_{2,\mathrm{head}}=\dfrac{dL_2}{dT_2}(T_{02}-T_b)\)",
        "kid2_inductance_headroom_over_event_ratio": r"\(\Delta L_{2,\mathrm{head}}/\Delta L_{1,\mathrm{event}}\)",
        "dfr_dT_Hz_per_K": r"\(\dfrac{df_r}{dT}=-\dfrac{\alpha_\phi f_r}{2Q_iT_0}\approx-\dfrac{\alpha_\phi f_0}{2Q_iT_0}\)",
        "dT_dE_K_per_J": r"\(\dfrac{dT}{dE}=\dfrac{1}{C}\)",
        "dphi_dE_rad_per_J": r"\(\dfrac{d\phi}{dE}=\dfrac{d\phi}{df}\dfrac{df_r}{dT}\dfrac{dT}{dE}\)",
        "deltafr_event_Hz": r"\(\Delta f_{r,\mathrm{event}}=\dfrac{df_r}{dT}\Delta T_{\mathrm{event}}\)",
        "deltaphi_event_rad": r"\(\Delta\phi_{\mathrm{event}}=\dfrac{d\phi}{dE}E_{\mathrm{Ho}}\)",
        "phonon_power_asd_device_W_per_rtHz": r"\(\sqrt{S_{P,\mathrm{ph}}}(T_0)=\sqrt{4k_BT_0^2G}\)",
        "phonon_temp_asd_device_K_per_rtHz": r"\(\sqrt{S_{T,\mathrm{ph}}}(T_0)=\dfrac{\sqrt{S_{P,\mathrm{ph}}}(T_0)}{G}\)",
        "phonon_energy_asd_device_J_per_rtHz": r"\(\sqrt{S_{E,\mathrm{ph}}}(T_0)=C\sqrt{S_{T,\mathrm{ph}}}(T_0)\)",
        "asd_phi_phonon_simple_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{simple}}(T_0)=\left|\dfrac{d\phi}{dE}\right|\sqrt{S_{E,\mathrm{ph}}}(T_0)\)",
        "thermal_energy_fluct_rms_J": r"\(\sigma_{E,\mathrm{th}}=\sqrt{k_BT_0^2C}\)",
        "thermal_energy_fluct_rms_eV": r"\(\sigma_{E,\mathrm{th,eV}}=\sigma_{E,\mathrm{th}}/q_e\)",
        "ho_decay_energy_eV": r"\(E_{\mathrm{Ho,eV}}=\dfrac{E_{\mathrm{Ho}}}{q_e}\)",
        "tau_th_s": r"\(\tau_{\mathrm{th}}=\dfrac{C}{G}\)",
        "nep_sufficient_frequency_hz": r"\(f_{\mathrm{NEP,suff}}=\max\{f:\sigma_E^{\mathrm{hi\to lo}}(f)\le(1+p_{\mathrm{NEP,suff}}/100)\sigma_{E,\mathrm{full}}\}\)",
        "nep_sufficient_time_s": r"\(\tau_{\mathrm{NEP,suff}}=\dfrac{1}{f_{\mathrm{NEP,suff}}}\)",
        "tau_target_from_rate_s": r"\(\tau_{\mathrm{target}}=\dfrac{P_{\mathrm{pileup,max}}}{R}\)",
        "tau_error_fraction": r"\(\epsilon_{\tau}=0\)",
        "pileup_probability_max": r"\(P_{\mathrm{pileup,reject}}=1-e^{-2R\tau_{\mathrm{NEP,suff}}}\)",
        "tau_res_s": r"\(\tau_{\mathrm{res}}=\dfrac{Q_r}{\pi f_0}\)",
        "tau_ratio_res_over_th": r"\(\dfrac{\tau_{\mathrm{res}}}{\tau_{\mathrm{th}}}\)",
        "core_rule1_left_ratio": r"\(\dfrac{\tau_{qp}}{\tau_{\mathrm{res}}}\)",
        "core_rule1_right_ratio": r"\(\dfrac{\tau_{\mathrm{res}}}{\tau_{\mathrm{th}}}\)",
        "core_rule1_ok": r"\(\text{Pass if } \tau_{qp}/\tau_{\mathrm{res}}<0.1\)",
        "core_rule2_ratio": r"\(\dfrac{\tau_{\mathrm{res}}}{\tau_{\mathrm{th}}}\)",
        "core_rule2_ok": r"\(\text{Pass if } \tau_{\mathrm{res}}/\tau_{\mathrm{th}}<1\)",
        "core_rule3_ok": r"\(\text{Pass if } \sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{simple}}>\sqrt{S_{\phi,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "core_rule4_ok": r"\(\text{Pass if } P_{0,x=0} \lt P_{\mathrm{bif}}\)",
        "core_rule5_ok": r"\(\text{Pass if } P_{0,x=0}/P_{\mathrm{bif}} \gt 0.5\)",
        "core_rule6_ok": r"\(\text{Pass if } P_{\mathrm{bif,min}}^{\mathrm{typ}} \le P_{\mathrm{bif,dBm}} \le P_{\mathrm{bif,max}}^{\mathrm{typ}}\)",
        "core_rule7_ok": r"\(\text{Pass if } \Delta T_{\mathrm{event}} \gt 1\,\mathrm{mK}\)",
        "core_rule8_ok": r"\(\text{Pass if } \Delta T_{\mathrm{event}} \lt 100\,\mathrm{mK}\)",
        "core_rule9_ok": r"\(\text{Pass if } \sigma_{E,\mathrm{th,eV}} \lt \sigma_{E,\mathrm{target}}\)",
        "core_rule10_ok": r"\(\text{Pass if } \Delta T_{2,\mathrm{head}} \ge \dfrac{\alpha_{\phi1}R_1}{\alpha_{\phi2}R_2}\Delta T_{1,\mathrm{event}}\ \text{(same-}T\text{ alpha assumption)}\)",
        "core_rule11_ok": r"\(\text{Pass if } T_0+\Delta T_{\mathrm{event}} \lt T_c\)",
        "core_rule12_ok": r"\(\text{Pass if } \Re[\lambda_i(M_t)]<0,\ \forall i\)",
        "core_rule13_ok": r"\(\text{Pass if } P_{\mathrm{pileup,reject}} < 0.5\)",
        "core_rule14_ok": r"\(\text{Pass if } |\mathrm{NEP}_{\phi}(0)/\mathrm{NEP}_{\phi,\mathrm{ph}}(0)-1|\le 0.01\)",
        "core_rule15_ok": r"\(\text{Pass if } P_{0,x=0}/P_{\mathrm{bif}}<1\)",
        "phonon_power_rms_W": r"\(P_{\mathrm{ph,RMS}}=\sqrt{4k_BT_b^2G}\)",
        "L_geo_H": r"\(L_g\approx \mu_0\ell\left[\ln\!\left(\dfrac{2\ell}{w}\right)+0.5\right]\)",
        "L_total_H": r"\(L_{\mathrm{tot}}=\dfrac{L_g}{1-\alpha_k}\)",
        "C_res_F": r"\(C_{\mathrm{res}}=\dfrac{1}{(2\pi f_0)^2L_{\mathrm{tot}}}\)",
        "Z0_res_Ohm": r"\(Z_0=(2\pi f_0)L_{\mathrm{tot}}\)",
        "Z0_readout_Ohm": r"\(Z_{\mathrm{readout}}=50\,\Omega\)",
        "readout_circle_radius_V": r"\(r_V=\sqrt{P_0Z_{\mathrm{readout}}}/2\)",
        "R0_Ohm": r"\(R_0=\dfrac{Z_0}{Q_r}\)",
        "Pg_W": r"\(P_g=\left(P_g/P_{\mathrm{bif}}\right)_{\mathrm{target}}\,P_{\mathrm{bif}}\)",
        "pg_to_p0_factor": r"\(P_0/P_g=\dfrac{1}{2}\dfrac{1}{1+4Q_r^2x^2}\dfrac{4Q_cQ_i}{(Q_c+Q_i)^2}\)",
        "P0_W": r"\(P_0=\left(P_0/P_g\right)P_g\)",
        "P0_undetuned_W": r"\(P_{0,x=0}=\dfrac{1}{2}\dfrac{4Q_cQ_i}{(Q_c+Q_i)^2}P_g\)",
        "Qc": r"\(Q_c=\left(\dfrac{1}{Q_r}-\dfrac{1}{Q_i}\right)^{-1}\)",
        "p_bifurcation_W": r"\(P_{\mathrm{bif}}=\dfrac{Q_c\omega_0E_*}{2Q_r^3}\)",
        "p_bifurcation_dBm": r"\(P_{\mathrm{bif,dBm}}=10\log_{10}(P_{\mathrm{bif}}/1\,\mathrm{mW})\)",
        "bifurcation_power_ratio": r"\(\dfrac{P_{0,x=0}}{P_{\mathrm{bif}}}\)",
        "delta_J": r"\(\Delta=1.764\,k_BT_c\)",
        "eqp_J": r"\(E_{qp}=\Delta\)",
        "x": r"\(x=\delta f/f_0\)",
        "johnson_voltage_rms_V": r"\(V_e=\sqrt{4k_BT_0R_0}\)",
        "johnson_sv_V2_per_Hz": r"\(S_{V,J}=4k_BT_0R_0\)",
        "amplifier_sideband_voltage_asd_V_per_rtHz": r"\(V_A=\sqrt{k_BT_AZ_{\mathrm{readout}}}\)",
        "amplifier_normalized_asd_per_rtHz": r"\(\sigma_{\mathrm{amp}}=\sqrt{8k_BT_A/P_0}\)",
        "M_e": r"\(M_e=\sqrt{\dfrac{E_{qp}}{k_BT_0}}\)",
        "N_J_scale": r"\(N_{J,\mathrm{scale}}=\sqrt{\dfrac{16k_BT_0}{P_0}}\)",
        "N_J_thermal_scale": r"\(N_{J,\mathrm{th}}=\sqrt{4k_BT_0P_0}\)",
        "mt_eig1_per_s": r"\(\lambda_1(M_t)\)",
        "mt_eig2_per_s": r"\(\lambda_2(M_t)\)",
        "mt_eig3_per_s": r"\(\lambda_3(M_t)\)",
        "mt_eig4_per_s": r"\(\lambda_4(M_t)\)",
        "mt_max_real_part_per_s": r"\(\max\Re[\lambda(M_t)]\)",
        "mt_stable": r"\(1\ \mathrm{if}\ \Re[\lambda_i(M_t)]<0\ \forall i\)",
        "mt_pulse_shortening_ratio": r"\(\rho_{\mathrm{short}}=\dfrac{G/C}{\min_i\left(-\Re[\lambda_i(M_t)]\right)}\ \mathrm{if\ stable}\)",
        "passive_tls_iq_transfer_phi_100hz_abs": r"\(|H_{\mathrm{TLS,passive}}|=\left|[M_{\mathrm{passive}}^{-1}(0,i)^T]_{\phi}\right|\)",
        "tls_iq_source_asd_100hz_per_rtHz": r"\(\sqrt{S_{N,\mathrm{TLS}}}(100\,\mathrm{Hz})=\dfrac{4\sqrt{S_{V,\mathrm{USB},\mathrm{TLS}}}(100\,\mathrm{Hz})}{I_0R_0}\)",
        "sphi_johnson_full_per_hz": r"\(S_{\phi,J}^{\mathrm{full}}=\left|[Y_{J,A}]_{\phi}\right|^2+\left|[Y_{J,\phi}]_{\phi}\right|^2\)",
        "sphi_tls_per_hz": r"\(S_{\phi,\mathrm{TLS}}=\left|[Y_{\mathrm{TLS}}]_{\phi}\right|^2\)",
        "sphi_amplifier_per_hz": r"\(S_{\phi,\mathrm{amp}}=\left|[Y_{\mathrm{amp},A}]_{\phi}\right|^2+\left|[Y_{\mathrm{amp},\phi}]_{\phi}\right|^2\)",
        "asd_phi_amplifier_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{amp}}}\)",
        "asd_phi_tls_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{TLS}}}\)",
        "asd_phi_tls_100hz_model_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{TLS}}}(100\,\mathrm{Hz})=f_0\left|d\phi/df\right|_{\delta f}\sqrt{\left(S_f/f_0^2\right)_{\mathrm{TLS}}(100\,\mathrm{Hz})}\)",
        "asd_phi_phonon_full_per_rtHz": r"\(\sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{full}}=\left|[Y_{ph}]_{\phi}\right|\)",
        "dphi_df_detuning_per_hz": r"\(\left|d\phi/df\right|_{x}\approx\dfrac{4Q_r/f_0}{1+(2Q_rx)^2}\)",
        "sf_over_f0sq_johnson_full": r"\(\left(S_f/f_0^2\right)_{J,\mathrm{full}}=\dfrac{S_{\phi,J}^{\mathrm{full}}}{f_0^2\left|d\phi/df\right|_{\delta f}^2}\)",
        "sf_over_f0sq_johnson_simple": r"\(\left(S_f/f_0^2\right)_{J,\mathrm{simple}}=\dfrac{S_{\phi,J}^{\mathrm{full}}}{(4Q_r)^2}\)",
        "m_phonon_over_johnson_phi": r"\(M_{\mathrm{ph}/J,\phi}=\dfrac{|[Y_{ph}]_{\phi}|}{|[Y_{J,\phi}]_{\phi}|}\)",
        "m_phonon_over_tls_phi": r"\(M_{\mathrm{ph}/\mathrm{TLS},\phi}=\dfrac{\sqrt{S_{\phi,\mathrm{ph}}}^{\,\mathrm{simple}}}{\sqrt{S_{\phi,\mathrm{TLS}}}(100\,\mathrm{Hz})}\)",
        "sf_over_f0sq_tls_model": r"\(\left(S_f/f_0^2\right)_{\mathrm{TLS}}(\nu)=\dfrac{1}{4}S_{\delta C/C,\mathrm{TLS}}(\nu)\)",
        "sf_over_f0sq_tls_1hz": r"\(\left(S_f/f_0^2\right)_{\mathrm{TLS},1\mathrm{Hz}}=\dfrac{1}{4}S_{\delta C/C,\mathrm{TLS}}(1\,\mathrm{Hz})\)",
        "I0_rms_A": r"\(I_0=\sqrt{P_0/R_0}\)",
        "s_deltaC_tls_1hz_F2_per_Hz": r"\(S_{\delta C,\mathrm{TLS}}(1\,\mathrm{Hz})=4C_0^2\left(S_f/f_0^2\right)_{\mathrm{TLS},1\mathrm{Hz}}\)",
        "asd_deltaC_tls_1hz_F_per_rtHz": r"\(\sqrt{S_{\delta C,\mathrm{TLS}}}(1\,\mathrm{Hz})\)",
        "s_deltaC_over_C_tls_100hz_per_Hz": r"\(S_{\delta C/C,\mathrm{TLS}}(100\,\mathrm{Hz})=4\left(S_f/f_0^2\right)_{\mathrm{TLS}}(100\,\mathrm{Hz})\)",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": r"\(\sqrt{S_{\delta C/C,\mathrm{TLS}}}(100\,\mathrm{Hz})\)",
        "sv_usb_tls_1hz_V2_per_Hz": r"\(S_{V,\mathrm{USB},\mathrm{TLS}}(1\,\mathrm{Hz})=\dfrac{I_0^2}{\omega_0^2C_0^2}\left(S_f/f_0^2\right)_{\mathrm{TLS},1\mathrm{Hz}}\)",
        "asd_v_usb_tls_1hz_V_per_rtHz": r"\(\sqrt{S_{V,\mathrm{USB},\mathrm{TLS}}}(1\,\mathrm{Hz})\)",
        "sv_usb_johnson_V2_per_Hz": r"\(S_{V,\mathrm{USB},J}=\dfrac{1}{2}S_{V,J}\)",
        "asd_v_usb_johnson_V_per_rtHz": r"\(\sqrt{S_{V,\mathrm{USB},J}}\)",
        "m_usb_tls_over_johnson_1hz": r"\(M_{\mathrm{USB},\mathrm{TLS}/J}(1\,\mathrm{Hz})=\dfrac{\sqrt{S_{V,\mathrm{USB},\mathrm{TLS}}(1\,\mathrm{Hz})}}{\sqrt{S_{V,\mathrm{USB},J}}}\)",
    }
    # Enforce: every non-constant symbol in an output formula must correspond
    # to an input or output quantity key in this model.
    formula_dependencies = {
        "au_number_density_per_m3": set(),
        "ho_number_density_per_m3": {"ho_in_au_atomic_fraction", "au_number_density_per_m3"},
        "ho_decay_constant_per_s": set(),
        "ho_activity_per_m3_Hz": {"ho_decay_constant_per_s", "ho_number_density_per_m3"},
        "absorber_volume_m3": {"count_rate_Hz", "ho_activity_per_m3_Hz"},
        "absorber_edge_m": {"absorber_volume_m3"},
        "absorber_length_m": {"absorber_edge_m"},
        "absorber_width_m": {"absorber_edge_m"},
        "absorber_thickness_m": {"absorber_edge_m"},
        "absorber_island_length_m": {"membrane_length_m"},
        "absorber_island_width_m": {"membrane_width_m"},
        "absorber_island_area_m2": {"absorber_island_length_m", "absorber_island_width_m"},
        "membrane_volume_m3": {"membrane_length_m", "membrane_width_m", "membrane_thickness_m"},
        "membrane_heat_capacity_J_per_K": {"cv_absorber_J_per_m3K", "membrane_volume_m3"},
        "membrane_heat_capacity_eV_per_mK": {"membrane_heat_capacity_J_per_K"},
        "membrane_length_m": {"absorber_length_m", "kid_length_m", "membrane_margin_m"},
        "membrane_width_m": {"absorber_width_m", "kid_width_m", "membrane_margin_m"},
        "membrane_span_m": {"membrane_length_m", "membrane_width_m"},
        "leg_length_m": {"leg_count", "kappa_leg_W_per_mK", "leg_width_m", "leg_thickness_m", "G_W_per_K"},
        "leg_thickness_m": {"membrane_thickness_m"},
        "C_J_per_K": {"cv_absorber_J_per_m3K", "absorber_volume_m3"},
        "C_eV_per_mK": {"C_J_per_K"},
        "C_ho_eV_per_mK": {"C_J_per_K"},
        "G_W_per_K": {"P0_W", "deltaT_abs_over_bath_setpoint_K"},
        "deltaT_abs_over_bath_K": {"P0_W", "G_W_per_K"},
        "tbath_from_link_K": {"T0_K", "deltaT_abs_over_bath_K"},
        "deltaT_event_full_absorption_K": {"ho_decay_energy_J", "C_J_per_K"},
        "kid2_thermal_headroom_K": {"T02_K", "Tb_K"},
        "kid2_thermal_headroom_over_event_ratio": {
            "kid2_thermal_headroom_K",
            "deltaT_event_full_absorption_K",
        },
        "dL1_dT_H_per_K": {"alpha_phi", "L_total_H", "T0_K"},
        "dL2_dT_H_per_K": {"alpha_phi2", "series_L2_H", "T02_K"},
        "deltaL1_event_H": {"dL1_dT_H_per_K", "deltaT_event_full_absorption_K"},
        "deltaL2_compensation_headroom_H": {"dL2_dT_H_per_K", "kid2_thermal_headroom_K"},
        "kid2_inductance_headroom_over_event_ratio": {
            "deltaL2_compensation_headroom_H",
            "deltaL1_event_H",
        },
        "dfr_dT_Hz_per_K": {"alpha_phi", "f0_Hz", "Qi", "T0_K"},
        "dT_dE_K_per_J": {"C_J_per_K"},
        "dphi_dE_rad_per_J": {"dphi_df_detuning_per_hz", "dfr_dT_Hz_per_K", "dT_dE_K_per_J"},
        "deltafr_event_Hz": {"dfr_dT_Hz_per_K", "deltaT_event_full_absorption_K"},
        "deltaphi_event_rad": {"dphi_dE_rad_per_J", "ho_decay_energy_J"},
        "phonon_power_asd_device_W_per_rtHz": {"T0_K", "G_W_per_K"},
        "phonon_temp_asd_device_K_per_rtHz": {"phonon_power_asd_device_W_per_rtHz", "G_W_per_K"},
        "phonon_energy_asd_device_J_per_rtHz": {"C_J_per_K", "phonon_temp_asd_device_K_per_rtHz"},
        "asd_phi_phonon_simple_per_rtHz": {"dphi_dE_rad_per_J", "phonon_energy_asd_device_J_per_rtHz"},
        "thermal_energy_fluct_rms_J": {"T0_K", "C_J_per_K"},
        "thermal_energy_fluct_rms_eV": {"thermal_energy_fluct_rms_J"},
        "ho_decay_energy_eV": {"ho_decay_energy_J"},
        "tau_th_s": {"C_J_per_K", "G_W_per_K"},
        "nep_sufficient_frequency_hz": {"nep_sufficiency_percent", "nep_phi_total_W_per_rtHz"},
        "nep_sufficient_time_s": {"nep_sufficient_frequency_hz"},
        "tau_target_from_rate_s": {"count_rate_Hz"},
        "tau_error_fraction": set(),
        "tau_res_s": {"Qr", "f0_Hz"},
        "tau_ratio_res_over_th": {"tau_res_s", "tau_th_s"},
        "phonon_power_rms_W": {"Tb_K", "G_W_per_K"},
        "L_geo_H": {"kid_trace_length_m", "kid_trace_width_m"},
        "L_total_H": {"L_geo_H", "kinetic_inductance_fraction"},
        "C_res_F": {"f0_Hz", "L_total_H"},
        "Z0_res_Ohm": {"f0_Hz", "L_total_H"},
        "R0_Ohm": {"Z0_res_Ohm", "Qr"},
        "Pg_W": {"pg_drive_dBm"},
        "pg_to_p0_factor": {"Qr", "Qi", "Qc", "x"},
        "P0_W": {"Pg_W", "pg_to_p0_factor"},
        "P0_undetuned_W": {"Pg_W", "Qi", "Qc"},
        "Qc": set(),
        "p_bifurcation_W": {"Qc", "f0_Hz", "bifurcation_energy_scale_J", "Qr"},
        "p_bifurcation_dBm": {"p_bifurcation_W"},
        "bifurcation_power_ratio": {"P0_undetuned_W", "p_bifurcation_W"},
        "delta_J": {"Tc_K"},
        "eqp_J": {"delta_J"},
        "johnson_voltage_rms_V": {"T0_K", "R0_Ohm"},
        "johnson_sv_V2_per_Hz": {"T0_K", "R0_Ohm"},
        "M_e": {"eqp_J", "T0_K"},
        "N_J_scale": {"T0_K", "P0_W"},
        "N_J_thermal_scale": {"T0_K", "P0_W"},
        "mt_eig1_per_s": set(),
        "mt_eig2_per_s": set(),
        "mt_eig3_per_s": set(),
        "mt_eig4_per_s": set(),
        "mt_max_real_part_per_s": set(),
        "mt_stable": set(),
        "mt_pulse_shortening_ratio": {
            "mt_stable",
            "mt_eig1_per_s",
            "mt_eig2_per_s",
            "mt_eig3_per_s",
            "mt_eig4_per_s",
            "G_W_per_K",
            "C_J_per_K",
        },
        "core_rule3_ok": {"asd_phi_phonon_simple_per_rtHz", "asd_phi_tls_100hz_model_per_rtHz"},
        "core_rule4_ok": {"P0_undetuned_W", "p_bifurcation_W"},
        "core_rule5_ok": {"bifurcation_power_ratio"},
        "core_rule6_ok": {"pbif_typical_min_dBm", "p_bifurcation_dBm", "pbif_typical_max_dBm"},
        "core_rule7_ok": {"deltaT_event_full_absorption_K"},
        "core_rule8_ok": {"deltaT_event_full_absorption_K"},
        "core_rule9_ok": {"thermal_energy_fluct_rms_eV", "thermal_energy_resolution_target_eV"},
        "core_rule10_ok": {"deltaL2_compensation_headroom_H", "deltaL1_event_H"},
        "core_rule11_ok": {"event_peak_temperature_K", "Tc_K"},
        "core_rule12_ok": {"mt_stable"},
        "core_rule13_ok": {"pileup_probability_max"},
        "core_rule14_ok": {"nep_phi_0hz_over_phonon_ratio"},
        "core_rule15_ok": {"p0_over_pbif_target"},
        "sphi_johnson_full_per_hz": set(),
        "sphi_tls_per_hz": set(),
        "asd_phi_tls_per_rtHz": {"sphi_tls_per_hz"},
        "asd_phi_tls_100hz_model_per_rtHz": {
            "tls_phi_asd_100hz_per_rtHz",
        },
        "passive_tls_iq_transfer_phi_100hz_abs": {"f0_Hz", "Qr", "Qi", "detuning_Hz"},
        "tls_iq_source_asd_100hz_per_rtHz": {
            "I0_rms_A",
            "R0_Ohm",
            "f0_Hz",
            "C_res_F",
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
        },
        "asd_phi_phonon_full_per_rtHz": set(),
        "dphi_df_detuning_per_hz": {"Qr", "f0_Hz", "detuning_Hz"},
        "x": {"detuning_Hz", "f0_Hz"},
        "sf_over_f0sq_johnson_full": {"sphi_johnson_full_per_hz", "f0_Hz", "dphi_df_detuning_per_hz"},
        "sf_over_f0sq_johnson_simple": {"sphi_johnson_full_per_hz", "Qr"},
        "m_phonon_over_johnson_phi": set(),
        "m_phonon_over_tls_phi": {"asd_phi_phonon_simple_per_rtHz", "asd_phi_tls_100hz_model_per_rtHz"},
        "sf_over_f0sq_tls_model": {
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
            "tls_beta",
        },
        "sf_over_f0sq_tls_1hz": {
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
            "tls_beta",
        },
        "I0_rms_A": {"P0_W", "R0_Ohm"},
        "s_deltaC_tls_1hz_F2_per_Hz": {"C_res_F", "sf_over_f0sq_tls_1hz"},
        "asd_deltaC_tls_1hz_F_per_rtHz": {"s_deltaC_tls_1hz_F2_per_Hz"},
        "s_deltaC_over_C_tls_100hz_per_Hz": {"asd_deltaC_over_C_tls_100hz_per_rtHz"},
        "asd_deltaC_over_C_tls_100hz_per_rtHz": {"s_deltaC_over_C_tls_100hz_per_Hz"},
        "sv_usb_tls_1hz_V2_per_Hz": {"I0_rms_A", "f0_Hz", "C_res_F", "sf_over_f0sq_tls_1hz"},
        "asd_v_usb_tls_1hz_V_per_rtHz": {"sv_usb_tls_1hz_V2_per_Hz"},
        "sv_usb_johnson_V2_per_Hz": {"johnson_sv_V2_per_Hz"},
        "asd_v_usb_johnson_V_per_rtHz": {"sv_usb_johnson_V2_per_Hz"},
        "m_usb_tls_over_johnson_1hz": {"asd_v_usb_tls_1hz_V_per_rtHz", "asd_v_usb_johnson_V_per_rtHz"},
        "P1_W": {"P0_W", "R0_Ohm"},
        "P2_W": {"P0_W", "series_R2_Ohm"},
        "heater2_dc_power_W": {"G2_W_per_K", "T02_K", "Tb_K", "P2_W"},
        "heater2_dc_power_dBm": {"heater2_dc_power_W"},
    }
    wiki_links = {
        "au_number_density_per_m3": "project.html#source-activity-formulas",
        "ho_number_density_per_m3": "project.html#source-activity-formulas",
        "ho_decay_constant_per_s": "project.html#source-activity-formulas",
        "ho_activity_per_m3_Hz": "project.html#source-activity-formulas",
        "absorber_volume_m3": "project.html#source-activity-formulas",
        "absorber_edge_m": "project.html#source-activity-formulas",
        "absorber_length_m": "project.html#source-activity-formulas",
        "absorber_width_m": "project.html#source-activity-formulas",
        "absorber_thickness_m": "project.html#source-activity-formulas",
        "absorber_island_length_m": "project.html#membrane-geometry-formulas",
        "absorber_island_width_m": "project.html#membrane-geometry-formulas",
        "absorber_island_area_m2": "project.html#membrane-geometry-formulas",
        "membrane_volume_m3": "project.html#membrane-geometry-formulas",
        "membrane_heat_capacity_J_per_K": "physics.html#thermal-derived-formulas",
        "membrane_heat_capacity_eV_per_mK": "physics.html#thermal-derived-formulas",
        "membrane_length_m": "project.html#membrane-geometry-formulas",
        "membrane_width_m": "project.html#membrane-geometry-formulas",
        "membrane_span_m": "project.html#membrane-geometry-formulas",
        "leg_length_m": "physics.html#leg-geometry-formulas",
        "leg_thickness_m": "physics.html#leg-geometry-formulas",
        "C_J_per_K": "physics.html#thermal-derived-formulas",
        "C_eV_per_mK": "physics.html#thermal-derived-formulas",
        "C_ho_eV_per_mK": "physics.html#thermal-derived-formulas",
        "G_W_per_K": "physics.html#thermal-derived-formulas",
        "deltaT_abs_over_bath_K": "physics.html#thermal-derived-formulas",
        "tbath_from_link_K": "physics.html#thermal-derived-formulas",
        "deltaT_event_full_absorption_K": "physics.html#thermal-derived-formulas",
        "kid2_thermal_headroom_K": "project.html#core-rule-checks",
        "kid2_thermal_headroom_over_event_ratio": "project.html#core-rule-checks",
        "dL1_dT_H_per_K": "project.html#core-rule-checks",
        "dL2_dT_H_per_K": "project.html#core-rule-checks",
        "deltaL1_event_H": "project.html#core-rule-checks",
        "deltaL2_compensation_headroom_H": "project.html#core-rule-checks",
        "kid2_inductance_headroom_over_event_ratio": "project.html#core-rule-checks",
        "dfr_dT_Hz_per_K": "simple-estimates.html",
        "dT_dE_K_per_J": "simple-estimates.html",
        "dphi_dE_rad_per_J": "simple-estimates.html",
        "deltafr_event_Hz": "simple-estimates.html",
        "deltaphi_event_rad": "simple-estimates.html",
        "phonon_power_asd_device_W_per_rtHz": "simple-estimates.html",
        "phonon_temp_asd_device_K_per_rtHz": "simple-estimates.html",
        "phonon_energy_asd_device_J_per_rtHz": "simple-estimates.html",
        "asd_phi_phonon_simple_per_rtHz": "simple-estimates.html",
        "thermal_energy_fluct_rms_J": "simple-estimates.html",
        "thermal_energy_fluct_rms_eV": "simple-estimates.html",
        "ho_decay_energy_eV": "physics.html#thermal-derived-formulas",
        "tau_th_s": "physics.html#thermal-derived-formulas",
        "nep_sufficient_frequency_hz": "nep-to-sigma.html",
        "nep_sufficient_time_s": "pileup-analysis.html",
        "tau_target_from_rate_s": "physics.html#thermal-derived-formulas",
        "tau_error_fraction": "physics.html#thermal-derived-formulas",
        "tau_res_s": "physics.html#resonator-derived-formulas",
        "tau_ratio_res_over_th": "physics.html#resonator-derived-formulas",
        "core_rule1_left_ratio": "project.html#core-rule-checks",
        "core_rule1_right_ratio": "project.html#core-rule-checks",
        "core_rule1_ok": "project.html#core-rule-checks",
        "core_rule2_ratio": "project.html#core-rule-checks",
        "core_rule2_ok": "project.html#core-rule-checks",
        "core_rule3_ok": "project.html#core-rule-checks",
        "core_rule4_ok": "project.html#core-rule-checks",
        "core_rule5_ok": "project.html#core-rule-checks",
        "core_rule6_ok": "project.html#core-rule-checks",
        "core_rule7_ok": "project.html#core-rule-checks",
        "core_rule8_ok": "project.html#core-rule-checks",
        "core_rule9_ok": "project.html#core-rule-checks",
        "core_rule10_ok": "project.html#core-rule-checks",
        "core_rule11_ok": "project.html#core-rule-checks",
        "core_rule12_ok": "project.html#core-rule-checks",
        "core_rule13_ok": "project.html#core-rule-checks",
        "core_rule14_ok": "project.html#core-rule-checks",
        "core_rule15_ok": "project.html#core-rule-checks",
        "phonon_power_rms_W": "noise-phonon.html#physical-expression",
        "L_geo_H": "theory.html#kid-lumped-formulas",
        "L_total_H": "theory.html#kid-lumped-formulas",
        "C_res_F": "theory.html#kid-lumped-formulas",
        "Z0_res_Ohm": "theory.html#kid-lumped-formulas",
        "R0_Ohm": "theory.html#kid-lumped-formulas",
        "Qc": "bifurcation-limit.html#bifurcation-model",
        "p_bifurcation_W": "bifurcation-limit.html#bifurcation-model",
        "p_bifurcation_dBm": "bifurcation-limit.html#operating-margin",
        "bifurcation_power_ratio": "bifurcation-limit.html#operating-margin",
        "Pg_W": "readout-power-mapping.html#equation-3-mapping",
        "pg_to_p0_factor": "readout-power-mapping.html#equation-3-mapping",
        "delta_J": "theory.html#gap-formulas",
        "eqp_J": "theory.html#gap-formulas",
        "johnson_voltage_rms_V": "noise-johnson.html#physical-expression",
        "johnson_sv_V2_per_Hz": "noise-johnson.html#physical-expression",
        "M_e": "noise-electronic.html#noise-vector-derivation-eq-19",
        "N_J_scale": "noise-johnson.html#normalized-vector-derivation-eq-18",
        "N_J_thermal_scale": "noise-johnson.html#normalized-vector-derivation-eq-18",
        "mt_eig1_per_s": "mt-stability.html#stability-criterion",
        "mt_eig2_per_s": "mt-stability.html#stability-criterion",
        "mt_eig3_per_s": "mt-stability.html#stability-criterion",
        "mt_eig4_per_s": "mt-stability.html#stability-criterion",
        "mt_max_real_part_per_s": "mt-stability.html#stability-criterion",
        "mt_stable": "mt-stability.html#stability-criterion",
        "mt_pulse_shortening_ratio": "mt-stability.html#pulse-shortening",
        "sphi_johnson_full_per_hz": "noise-johnson.html#role-in-nep-and-electronic-noise-scaling",
        "sphi_tls_per_hz": "noise-tls.html#use-in-total-budget",
        "asd_phi_tls_per_rtHz": "noise-tls.html#use-in-total-budget",
        "asd_phi_tls_100hz_model_per_rtHz": "project.html#core-rule-checks",
        "passive_tls_iq_transfer_phi_100hz_abs": "noise-tls.html",
        "tls_iq_source_asd_100hz_per_rtHz": "noise-tls.html",
        "dphi_df_detuning_per_hz": "noise-johnson-sf.html",
        "sf_over_f0sq_johnson_full": "noise-johnson-sf.html",
        "sf_over_f0sq_johnson_simple": "noise-johnson-sf.html",
        "m_phonon_over_johnson_phi": "noise-phonon.html#physical-expression",
        "m_phonon_over_tls_phi": "project.html#core-rule-checks",
        "sf_over_f0sq_tls_model": "noise-tls.html#design-scaling-and-geometry-controls",
        "sf_over_f0sq_tls_1hz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "I0_rms_A": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "s_deltaC_tls_1hz_F2_per_Hz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "asd_deltaC_tls_1hz_F_per_rtHz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "s_deltaC_over_C_tls_100hz_per_Hz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "asd_deltaC_over_C_tls_100hz_per_rtHz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "sv_usb_tls_1hz_V2_per_Hz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "asd_v_usb_tls_1hz_V_per_rtHz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "sv_usb_johnson_V2_per_Hz": "noise-johnson.html#sideband-vector-derivation-eq-17",
        "asd_v_usb_johnson_V_per_rtHz": "noise-johnson.html#sideband-vector-derivation-eq-17",
        "m_usb_tls_over_johnson_1hz": "tls-capacitor-to-phase.html#practical-c-to-usb-mapping",
        "P1_W": "m-matrix.html",
        "P2_W": "m-matrix.html",
        "heater2_dc_power_W": "m-matrix.html",
        "heater2_dc_power_dBm": "m-matrix.html",
    }

    vectors = {
        "N_ph": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_phonon()],
        "N_J_A": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_johnson_A()],
        "N_J_phi": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_johnson_phi()],
        "N_g_A": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_electronic_A()],
        "N_g_phi": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_electronic_phi()],
        "N_TLS_phi_example": [complex(v).real if complex(v).imag == 0 else [complex(v).real, complex(v).imag] for v in s.n_tls_phi_at_hz(1.0)],
    }
    vector_component_units = {
        "basis": ["r-channel", "phi-channel", "thermal1-channel", "thermal2-channel"],
        "N_ph": ["r-units", "phi-units", "W/Hz^(1/2)", "W/Hz^(1/2)"],
        "N_J_A": ["r-units", "phi-units", "W/Hz^(1/2)", "W/Hz^(1/2)"],
        "N_J_phi": ["r-units", "phi-units", "thermal-units", "thermal-units"],
        "N_g_A": ["r-units", "phi-units", "W/Hz^(1/2)", "W/Hz^(1/2)"],
        "N_g_phi": ["r-units", "phi-units", "thermal-units", "thermal-units"],
        "N_TLS_phi_example": ["r-units", "phi-units", "thermal-units", "thermal-units"],
    }
    m_1hz = s.m_matrix(1.0)
    m_1hz_serialized = [[[z.real, z.imag] for z in row] for row in m_1hz]
    mt_matrix_serialized = [[[z.real, z.imag] for z in row] for row in s.mt_matrix]
    mt_eigs_serialized = [[complex(v).real, complex(v).imag] for v in s.mt_eigenvalues]

    model_inputs = {k: getattr(s, k) for k in input_keys}
    model_outputs = {k: v for k, v in est.items() if k not in model_inputs}
    eigs = s.mt_eigenvalues_sorted
    model_outputs["mt_eig1_per_s"] = _fmt_complex(complex(eigs[0]))
    model_outputs["mt_eig2_per_s"] = _fmt_complex(complex(eigs[1]))
    model_outputs["mt_eig3_per_s"] = _fmt_complex(complex(eigs[2]))
    model_outputs["mt_eig4_per_s"] = _fmt_complex(complex(eigs[3])) if len(eigs) > 3 else "n/a"

    # Validate formula dependencies against available model quantities.
    allowed = set(model_inputs.keys()) | set(model_outputs.keys())
    missing_deps = {}
    for out_key, deps in formula_dependencies.items():
        missing = sorted(d for d in deps if d not in allowed)
        if missing:
            missing_deps[out_key] = missing
    if missing_deps:
        lines = [f"{k}: {', '.join(v)}" for k, v in missing_deps.items()]
        raise ValueError("Formula dependency check failed:\n" + "\n".join(lines))

    payload = {
        "notes": {
            "f0_Hz": "Nominal detector carrier frequency",
            "f_demod_Hz": "Demodulated analysis frequency",
            "detuning_widths": "Project input in resonator widths (fr/Qr)",
            "nep_sufficiency_percent": "Percent-above-full-integral threshold used for NEP sufficiency timing",
            "detuning_Hz": "Derived detuning from detuning_widths * f0_Hz / Qr",
            "complex_value_format": "[real, imag]",
        },
        "units": units,
        "model_inputs": model_inputs,
        "model_outputs": model_outputs,
        "vectors": vectors,
        "vector_component_units": vector_component_units,
        "M_matrix_1Hz": {
            "frequency_Hz": 1.0,
            "format": "[real, imag]",
            "rows": m_1hz_serialized,
        },
        "Mt_matrix": {
            "definition": "Mt = -inv(D1) @ D0; D0=M(0); D1=-i dM/domega",
            "format": "[real, imag]",
            "rows": mt_matrix_serialized,
        },
        "Mt_eigenvalues": {
            "format": "[real, imag]",
            "values": mt_eigs_serialized,
            "max_real_part_per_s": s.mt_max_real_part,
            "stable": s.mt_stable,
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_groups = {
        "Source Geometry": [
            "count_rate_Hz",
            "deltaT_abs_over_bath_setpoint_K",
            "kid_length_m",
            "kid_width_m",
            "membrane_margin_m",
            "leg_count",
            "leg_width_m",
            "cap_thickness_m",
            "membrane_thickness_m",
        ],
        "Setpoints": [
            "T0_K",
            "Tb_K",
            "p0_over_pbif_target",
            "f0_Hz",
            "f_demod_Hz",
            "detuning_widths",
            "detuning_pid_gain_Hz_per_rad",
            "detuning_pid_integrator_time_s",
            "detuning_pid_derivative_time_s",
            "detuning_pid_derivative_filter_factor",
            "pbif_typical_min_dBm",
            "pbif_typical_max_dBm",
            "thermal_energy_resolution_target_eV",
        ],
        "Nonlinearity": [
            "bifurcation_energy_scale_J",
        ],
        "KID Properties": [
            "Qr",
            "Qi",
            "tau_qp_s",
            "kinetic_inductance_fraction",
            "kid_trace_length_m",
            "kid_trace_width_m",
            "alpha_A",
            "alpha_phi",
            "beta_A",
            "beta_phi",
            "Tc_K",
        ],
        "Material and Activity": [
            "ho_in_au_atomic_fraction",
            "ho_decay_energy_J",
            "cv_absorber_J_per_m3K",
            "kappa_leg_W_per_mK",
        ],
        "Capacitor TLS": [
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
            "tls_beta",
            "passive_tls_iq_transfer_phi_100hz_abs",
            "tls_iq_source_asd_100hz_per_rtHz",
        ],
    }

    def _rows_for_keys(keys: list[str]) -> str:
        return "\n".join(
            f"<tr><td>{k}</td><td>{symbols.get(k,'')}</td><td><code>{_fmt(float(model_inputs[k]))}</code></td><td>{units.get(k,'')}</td></tr>"
            for k in keys
            if k in model_inputs
        )

    input_sections = "\n".join(
        f"""
    <section class="card">
      <h3>{group_name}</h3>
      <table>
        <tr><th>Quantity</th><th>Symbol</th><th>Value</th><th>Units</th></tr>
        {_rows_for_keys(keys)}
      </table>
    </section>
    """
        for group_name, keys in input_groups.items()
    )
    output_groups = {
        "Simple Estimates": [
            "dfr_dT_Hz_per_K",
            "dT_dE_K_per_J",
            "dphi_dE_rad_per_J",
            "deltafr_event_Hz",
            "deltaphi_event_rad",
            "sf_over_f0sq_johnson_simple",
            "phonon_power_asd_device_W_per_rtHz",
            "asd_phi_phonon_simple_per_rtHz",
            "thermal_energy_fluct_rms_J",
            "thermal_energy_fluct_rms_eV",
            "p_bifurcation_dBm",
        ],
        "Derived Geometry": [
            "au_number_density_per_m3",
            "ho_number_density_per_m3",
            "ho_decay_constant_per_s",
            "ho_activity_per_m3_Hz",
            "absorber_volume_m3",
            "absorber_edge_m",
            "absorber_length_m",
            "absorber_width_m",
            "absorber_thickness_m",
            "absorber_island_length_m",
            "absorber_island_width_m",
            "absorber_island_area_m2",
            "membrane_volume_m3",
            "membrane_length_m",
            "membrane_width_m",
            "membrane_span_m",
            "leg_length_m",
            "leg_thickness_m",
        ],
        "Derived Thermal": [
            "C_eV_per_mK",
            "C_ho_eV_per_mK",
            "G_W_per_K",
            "deltaT_abs_over_bath_K",
            "tbath_from_link_K",
            "deltaT_event_full_absorption_K",
            "ho_decay_energy_eV",
            "tau_th_s",
            "tau_target_from_rate_s",
            "tau_error_fraction",
            "tau_res_s",
            "tau_ratio_res_over_th",
            "phonon_power_rms_W",
            "membrane_heat_capacity_eV_per_mK",
        ],
        "Derived KID Electrical": [
            "f0_Hz",
            "detuning_widths",
            "f_demod_Hz",
            "L_geo_H",
            "L_total_H",
            "C_res_F",
            "Z0_res_Ohm",
            "Z0_readout_Ohm",
            "readout_circle_radius_V",
            "R0_Ohm",
            "P0_W",
            "P0_undetuned_W",
            "Pg_W",
            "pg_to_p0_factor",
            "Qc",
            "Qr",
            "Qi",
            "alpha_A",
            "alpha_phi",
            "beta_A",
            "beta_phi",
            "Tc_K",
            "delta_J",
            "eqp_J",
            "johnson_voltage_rms_V",
            "johnson_sv_V2_per_Hz",
            "amplifier_noise_temperature_K",
            "amplifier_sideband_voltage_asd_V_per_rtHz",
            "amplifier_normalized_asd_per_rtHz",
            "N_J_scale",
            "N_J_thermal_scale",
            "detuning_pid_transfer_Hz_per_rad",
            "detuning_pid_transfer_dx_per_rad",
            "dx_response_to_event_power_at_f_demod",
        ],
        "Derived TLS": [
            "sphi_johnson_full_per_hz",
            "sphi_tls_per_hz",
            "asd_phi_tls_per_rtHz",
            "sphi_amplifier_per_hz",
            "asd_phi_amplifier_per_rtHz",
            "asd_phi_tls_100hz_model_per_rtHz",
            "passive_tls_iq_transfer_phi_100hz_abs",
            "tls_iq_source_asd_100hz_per_rtHz",
            "dphi_df_detuning_per_hz",
            "sf_over_f0sq_johnson_full",
            "sf_over_f0sq_johnson_simple",
            "sf_over_f0sq_tls_model",
            "sf_over_f0sq_tls_1hz",
            "I0_rms_A",
            "s_deltaC_tls_1hz_F2_per_Hz",
            "asd_deltaC_tls_1hz_F_per_rtHz",
            "s_deltaC_over_C_tls_100hz_per_Hz",
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
            "sv_usb_tls_1hz_V2_per_Hz",
            "asd_v_usb_tls_1hz_V_per_rtHz",
            "sv_usb_johnson_V2_per_Hz",
            "asd_v_usb_johnson_V_per_rtHz",
        ],
        "Noise M Ratios": [
            "M_e",
            "m_phonon_over_johnson_phi",
            "m_phonon_over_tls_phi",
            "m_usb_tls_over_johnson_1hz",
        ],
        "Design Verification": [
            "tau_res_s",
            "tau_th_s",
            "core_rule1_left_ratio",
            "core_rule1_right_ratio",
            "core_rule2_ratio",
            "thermal_energy_fluct_rms_eV",
            "deltaT_event_full_absorption_K",
            "event_peak_temperature_K",
            "dL1_dT_H_per_K",
            "deltaL1_event_H",
            "p_bifurcation_W",
            "mt_eig1_per_s",
            "mt_eig2_per_s",
            "mt_eig3_per_s",
            "mt_eig4_per_s",
            "mt_max_real_part_per_s",
            "bifurcation_power_ratio",
            "mt_pulse_shortening_ratio",
        ],
        "Pass/Fail Checks": [
            "core_rule1_ok",
            "core_rule2_ok",
            "core_rule3_ok",
            "core_rule4_ok",
            "core_rule5_ok",
            "core_rule6_ok",
            "core_rule7_ok",
            "core_rule8_ok",
            "core_rule9_ok",
            "core_rule10_ok",
            "core_rule11_ok",
            "core_rule12_ok",
            "core_rule13_ok",
            "core_rule14_ok",
            "core_rule15_ok",
        ],
    }

    def _output_rows_for_keys(keys: list[str], outputs: dict[str, object]) -> str:
        rows = []
        for k in keys:
            if k not in outputs:
                continue
            sym = symbols.get(k, "")
            if k in wiki_links:
                sym = f'<a href="{wiki_links[k]}">{sym}</a>'
            val = outputs[k]
            if k.endswith("_ok") or k == "mt_stable":
                val_s = _pf_html(float(val))
            else:
                val_s = _fmt(float(val)) if isinstance(val, (int, float)) else str(val)
            rows.append(
                f"<tr><td>{k}</td><td>{sym}</td><td>{formulas.get(k,'')}</td><td><code>{val_s}</code></td><td>{units.get(k,'')}</td></tr>"
            )
        return "\n".join(rows)

    output_sections = "\n".join(
        f"""
    <section class="card">
      <h3>{group_name}</h3>
      <table>
        <tr><th>Quantity</th><th>Symbol</th><th>Formula</th><th>Value</th><th>Units</th></tr>
        {_output_rows_for_keys(keys, model_outputs)}
      </table>
    </section>
    """
        for group_name, keys in output_groups.items()
    )

    html = rf"""<!doctype html>
<html lang="en"> 
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Python-Derived Estimates | TKID Neutrino Detector Project Wiki</title>
  <link rel="stylesheet" href="styles.css" />
  <style>
    .pass {{ color: #0b8a2a; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\(', '\\)'], ['$', '$']] }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <header class="hero">
    <h1>Python-Derived Estimates</h1>
    <p style="color:#000;opacity:1;">Values computed by python/sensor.py; page generated by python/generate_wiki_estimates.py.</p>
  </header>

  <nav class="nav">
    <a href="index.html">Home</a>
    <a href="project.html">Project</a>
    <a href="theory.html">Theory</a>
    <a href="noise-sources.html">Noise Sources</a>
    <a href="python-estimates.html" class="active">Python Estimates</a>
    <a href="mt-stability.html">Mt Stability</a>
  </nav>

  <main class="container">
    <section class="card">
      <h2>Nominal Settings</h2>
      <p>All values use the <strong>version 1 preset</strong> (\(R={_fmt(model_outputs['count_rate_Hz'])}\,\mathrm{{Hz}}\)) with nominal <strong>f0 = 1.0 GHz</strong>, demodulated band frequency <strong>0 Hz</strong>, detuning <strong>{_fmt(model_inputs['detuning_widths'])} widths</strong>, and derived detuning <strong>{_fmt(model_outputs['detuning_Hz'])} Hz</strong>.</p>
      <p>JSON source: <code>../python/outputs/wiki_estimates.json</code></p>
    </section>

    <section class="card">
      <h2>Model Inputs</h2>
    </section>
    {input_sections}

    <section class="card">
      <h2>Model Outputs</h2>
    </section>
    {output_sections}

  </main>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_HTML_TOP.write_text(html, encoding="utf-8")

    design_html = rf"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Design Parameters | TKID Neutrino Detector Docs</title>
  <link rel="stylesheet" href="styles.css" />
  <style>
    .pass {{ color: #0b8a2a; font-weight: 700; }}
    .fail {{ color: #b42318; font-weight: 700; }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\(', '\\)'], ['$', '$']] }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <header class="hero">
    <h1>Design Parameters</h1>
    <p>Auto-generated from <code>python/sensor.py</code> by <code>python/generate_wiki_estimates.py</code>.</p>
  </header>

  <nav class="nav">
    <a href="index.html">Home</a>
    <a href="project.html">Project</a>
    <a href="physics.html">Physics</a>
    <a href="theory.html">Theory</a>
    <a href="noise-sources.html">Noise Sources</a>
    <a href="signal-processing.html">Signal Processing</a>
    <a href="design.html" class="active">Design Parameters</a>
    <a href="python-estimates.html">Python Estimates</a>
    <a href="mt-stability.html">Mt Stability</a>
  </nav>

  <main class="container">
    <section class="card">
      <h2>Setpoints and Readout</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>\\(T_0\\)</td><td><code>{_fmt(model_inputs['T0_K'])}</code> K</td></tr>
        <tr><td>\\(T_b\\)</td><td><code>{_fmt(model_inputs['Tb_K'])}</code> K</td></tr>
        <tr><td>\\(f_0\\)</td><td><code>{_fmt(model_inputs['f0_Hz'])}</code> Hz</td></tr>
        <tr><td>Detuning</td><td><code>{_fmt(model_inputs['detuning_widths'])}</code> widths (<code>{_fmt(model_outputs['detuning_Hz'])}</code> Hz)</td></tr>
        <tr><td>Detuning PID gain \(K_f\)</td><td><code>{_fmt(model_inputs['detuning_pid_gain_Hz_per_rad'])}</code> Hz/rad</td></tr>
        <tr><td>Integrator time \(\tau_i\)</td><td><code>{_fmt(model_inputs['detuning_pid_integrator_time_s'])}</code> s</td></tr>
        <tr><td>Derivative time \(\tau_d\)</td><td><code>{_fmt(model_inputs['detuning_pid_derivative_time_s'])}</code> s</td></tr>
        <tr><td>Derivative filter \(N_d\)</td><td><code>{_fmt(model_inputs['detuning_pid_derivative_filter_factor'])}</code></td></tr>
        <tr><td>NEP sufficiency threshold</td><td><code>{_fmt(model_inputs['nep_sufficiency_percent'])}</code> %</td></tr>
        <tr><td>Demod frequency</td><td><code>{_fmt(model_inputs['f_demod_Hz'])}</code> Hz</td></tr>
        <tr><td>\\(P_{{0,x=0}}/P_{{bif}}\\)</td><td><code>{_fmt(model_outputs['p0_over_pbif_target'])}</code></td></tr>
        <tr><td>\\(P_g\\)</td><td><code>{_fmt(model_outputs['Pg_W'])}</code> W</td></tr>
        <tr><td>\\(P_0\\)</td><td><code>{_fmt(model_outputs['P0_W'])}</code> W</td></tr>
        <tr><td>\\(P_{{0,x=0}}\\)</td><td><code>{_fmt(model_outputs['P0_undetuned_W'])}</code> W</td></tr>
        <tr><td>\\(E_*\\)</td><td><code>{_fmt(model_inputs['bifurcation_energy_scale_J'])}</code> J</td></tr>
      </table>
    </section>

    <section class="card">
      <h2>Source and Geometry Inputs</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Count rate \\(R\\)</td><td><code>{_fmt(model_outputs['count_rate_Hz'])}</code> Hz</td></tr>
        <tr><td>Pileup rejection probability</td><td><code>{_fmt(model_outputs['pileup_probability_max'])}</code></td></tr>
        <tr><td>NEP sufficient frequency</td><td><code>{_fmt(model_outputs['nep_sufficient_frequency_hz'])}</code> Hz</td></tr>
        <tr><td>NEP sufficient time</td><td><code>{_fmt(model_outputs['nep_sufficient_time_s'])}</code> s</td></tr>
        <tr><td>Ho/Au atomic fraction \\(x_{{Ho/Au}}\\)</td><td><code>{_fmt(model_inputs['ho_in_au_atomic_fraction'])}</code></td></tr>
        <tr><td>Ho activity \\(A_{{Ho}}\\)</td><td><code>{_fmt(model_outputs['ho_activity_per_m3_Hz'])}</code> Hz/m^3</td></tr>
        <tr><td>Ho event energy \\(E_{{Ho}}\\)</td><td><code>{_fmt(model_inputs['ho_decay_energy_J'])}</code> J (<code>{_fmt(model_outputs['ho_decay_energy_eV'])}</code> eV)</td></tr>
        <tr><td>KID footprint \\(L_{{KID}},W_{{KID}}\\)</td><td><code>{_fmt(model_inputs['kid_length_m'])}</code> m, <code>{_fmt(model_inputs['kid_width_m'])}</code> m</td></tr>
        <tr><td>Membrane margin \\(\\Delta_{{mem}}\\)</td><td><code>{_fmt(model_inputs['membrane_margin_m'])}</code> m</td></tr>
        <tr><td>Leg count \\(N_{{leg}}\\)</td><td><code>{int(model_inputs['leg_count'])}</code></td></tr>
        <tr><td>Leg width \\(w_{{leg}}\\)</td><td><code>{_fmt(model_inputs['leg_width_m'])}</code> m</td></tr>
        <tr><td>Membrane thickness \\(t_{{mem}}\\)</td><td><code>{_fmt(model_inputs['membrane_thickness_m'])}</code> m</td></tr>
        <tr><td>Cap thickness \\(t_{{cap}}\\)</td><td><code>{_fmt(model_inputs['cap_thickness_m'])}</code> m</td></tr>
      </table>
      <p>Reference for percent-level alloy loading guidance: M. B. Sisti et al., <em>Specific Heat of Holmium in Gold and Silver at Low Temperatures</em> (see discussion of x_Ho >= 1% as a practical regime), <a href="https://arxiv.org/abs/1912.09354">arXiv:1912.09354</a>.</p>
      <p>Related pages: <a href="absorber-source.html">Absorber + Source</a>, <a href="idc-geometry.html">IDC Geometry</a>, <a href="thermal-conductance.html">Thermal Conductance</a>.</p>
      <p>Related page: <a href="frequency-multiplexing.html">Frequency + Mux</a>.</p>
    </section>

    <section class="card">
      <h2>Derived Thermal Nominals</h2>
      <table>
        <tr><th>Quantity</th><th>Value</th></tr>
        <tr><td>\\(C\\)</td><td><code>{_fmt(model_outputs['C_J_per_K'])}</code> J/K</td></tr>
        <tr><td>\\(C_{{eV/mK}}\\)</td><td><code>{_fmt(model_outputs['C_eV_per_mK'])}</code> eV/mK</td></tr>
        <tr><td>\\(C_{{Ho,eV/mK}}\\)</td><td><code>{_fmt(model_outputs['C_ho_eV_per_mK'])}</code> eV/mK</td></tr>
        <tr><td>\\(G\\)</td><td><code>{_fmt(model_outputs['G_W_per_K'])}</code> W/K</td></tr>
        <tr><td>\\(\\tau_{{th}}\\)</td><td><code>{_fmt(model_outputs['tau_th_s'])}</code> s</td></tr>
        <tr><td>\\(\\Delta T_{{event}}\\)</td><td><code>{_fmt(model_outputs['deltaT_event_full_absorption_K'])}</code> K</td></tr>
      </table>
    </section>

    <section class="card">
      <h2>Design Verification</h2>
      <table>
        <tr><th>Check</th><th>Value</th></tr>
        <tr><td>\\(\\tau_{{res}}\\)</td><td><code>{_fmt(model_outputs['tau_res_s'])}</code> s</td></tr>
        <tr><td>\\(\\tau_{{th}}\\)</td><td><code>{_fmt(model_outputs['tau_th_s'])}</code> s</td></tr>
        <tr><td><strong>Rule Statuses</strong></td><td><strong>Pass/Fail</strong></td></tr>
        <tr><td>Rule 1</td><td><code>{_pf_html(model_outputs['core_rule1_ok'])}</code></td></tr>
        <tr><td>Rule 2</td><td><code>{_pf_html(model_outputs['core_rule2_ok'])}</code></td></tr>
        <tr><td>Rule 3</td><td><code>{_pf_html(model_outputs['core_rule3_ok'])}</code></td></tr>
        <tr><td>Rule 4</td><td><code>{_pf_html(model_outputs['core_rule4_ok'])}</code></td></tr>
        <tr><td>Rule 5</td><td><code>{_pf_html(model_outputs['core_rule5_ok'])}</code></td></tr>
        <tr><td>Rule 6</td><td><code>{_pf_html(model_outputs['core_rule6_ok'])}</code></td></tr>
        <tr><td>Rule 7</td><td><code>{_pf_html(model_outputs['core_rule7_ok'])}</code></td></tr>
        <tr><td>Rule 8</td><td><code>{_pf_html(model_outputs['core_rule8_ok'])}</code></td></tr>
        <tr><td>Rule 9</td><td><code>{_pf_html(model_outputs['core_rule9_ok'])}</code></td></tr>
        <tr><td>Rule 10</td><td><code>{_pf_html(model_outputs['core_rule10_ok'])}</code></td></tr>
        <tr><td>Rule 11</td><td><code>{_pf_html(model_outputs['core_rule11_ok'])}</code></td></tr>
        <tr><td>Rule 12</td><td><code>{_pf_html(model_outputs['core_rule12_ok'])}</code></td></tr>
        <tr><td>Rule 13</td><td><code>{_pf_html(model_outputs['core_rule13_ok'])}</code></td></tr>
        <tr><td>Rule 14</td><td><code>{_pf_html(model_outputs['core_rule14_ok'])}</code></td></tr>
        <tr><td>Rule 15</td><td><code>{_pf_html(model_outputs['core_rule15_ok'])}</code></td></tr>
        <tr><td><strong>Bifurcation Metrics</strong></td><td></td></tr>
        <tr><td>\\(\\Delta T_{{event}}\\)</td><td><code>{_fmt(model_outputs['deltaT_event_full_absorption_K'])}</code> K</td></tr>
        <tr><td>\\(T_{{event,peak}}\\)</td><td><code>{_fmt(model_outputs['event_peak_temperature_K'])}</code> K</td></tr>
        <tr><td>\\(P_{{bif}}\\)</td><td><code>{_fmt(model_outputs['p_bifurcation_W'])}</code> W</td></tr>
        <tr><td>\\(P_{{bif,dBm}}\\)</td><td><code>{_fmt(model_outputs['p_bifurcation_dBm'])}</code> dBm</td></tr>
        <tr><td>\\(P_{{0,x=0}}/P_{{bif}}\\)</td><td><code>{_fmt(model_outputs['bifurcation_power_ratio'])}</code></td></tr>
        <tr><td>\\(\\max \\Re[\\lambda(M_t)]\\)</td><td><code>{_fmt(model_outputs['mt_max_real_part_per_s'])}</code> 1/s</td></tr>
        <tr><td>\\(\\rho_{{short}}\\)</td><td><code>{_fmt(model_outputs['mt_pulse_shortening_ratio'])}</code></td></tr>
      </table>
      <p>Stability details: <a href="mt-stability.html">Mt Stability</a>.</p>
    </section>
  </main>
</body>
</html>
"""
    OUT_DESIGN_HTML.write_text(design_html, encoding="utf-8")
    OUT_DESIGN_HTML_TOP.write_text(design_html, encoding="utf-8")


if __name__ == "__main__":
    main()

