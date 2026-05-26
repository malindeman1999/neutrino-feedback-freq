"""Pulse-sweep generation and frequency-domain optimal-filter analysis."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from math import ceil, pi
from typing import Any

import numpy as np

from sensor import J_PER_EV, Sensor


DATASET_FORMAT = "neutrino-pileup-phi-sweeps-v3"


def _time_constants_s(sensor: Sensor) -> tuple[float, float]:
    eigs = np.asarray(sensor.mt_eigenvalues, dtype=complex)
    stable = np.real(eigs) < 0.0
    if not np.any(stable):
        raise ValueError("pulse generation requires at least one stable decay mode")
    taus = -1.0 / np.real(eigs[stable])
    return float(np.min(taus)), float(np.max(taus))


def phase_noise_asd_rad_per_rtHz(sensor: Sensor, freqs_hz: np.ndarray) -> np.ndarray:
    """Total measured phase ASD on a one-sided frequency grid."""
    out = np.zeros_like(freqs_hz, dtype=float)
    for i, f_hz in enumerate(np.asarray(freqs_hz, dtype=float)):
        eval_hz = float(f_hz if f_hz > 0.0 else 1.0)
        y_j_a = sensor._propagate_noise_vector(sensor.n_johnson_A_1(), eval_hz)
        y_j_p = sensor._propagate_noise_vector(sensor.n_johnson_phi_1(), eval_hz)
        y_ph = sensor._propagate_noise_vector(sensor.n_phonon_1(), eval_hz)
        y_tls = sensor._propagate_noise_vector(sensor.n_tls_phi_at_hz(eval_hz), eval_hz)
        y_e_a = sensor._propagate_noise_vector(sensor.n_electronic_A_1(), eval_hz)
        y_e_p = sensor._propagate_noise_vector(sensor.n_electronic_phi_1(), eval_hz)
        y_amp_a = sensor.y_amplifier_A_at_hz(eval_hz)
        y_amp_p = sensor.y_amplifier_phi_at_hz(eval_hz)
        out[i] = np.sqrt(
            abs(y_j_a[1]) ** 2
            + abs(y_j_p[1]) ** 2
            + abs(y_ph[1]) ** 2
            + abs(y_tls[1]) ** 2
            + abs(y_e_a[1]) ** 2
            + abs(y_e_p[1]) ** 2
            + abs(y_amp_a[1]) ** 2
            + abs(y_amp_p[1]) ** 2
        )
    return out


def phase_energy_response_fft_rad_per_eV(sensor: Sensor, freqs_hz: np.ndarray) -> np.ndarray:
    """Fourier-domain phase response to an absorbed one-eV energy impulse."""
    src = (0.0 + 0.0j, 0.0 + 0.0j, sensor.event_power_fraction_kid1_clamped + 0.0j)
    return np.asarray(
        [sensor._solve_response_vector(src, float(f_hz))[1] * J_PER_EV for f_hz in freqs_hz],
        dtype=complex,
    )


def _noise_sweeps(
    asd_rad_per_rtHz: np.ndarray, n_samples: int, dt_s: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    spectra = np.zeros((count, asd_rad_per_rtHz.size), dtype=complex)
    if asd_rad_per_rtHz.size > 2:
        z = (rng.standard_normal((count, asd_rad_per_rtHz.size - 2)) + 1j * rng.standard_normal(
            (count, asd_rad_per_rtHz.size - 2)
        )) / np.sqrt(2.0)
        spectra[:, 1:-1] = z * asd_rad_per_rtHz[None, 1:-1] * np.sqrt(n_samples / (2.0 * dt_s))
    if n_samples % 2 == 0:
        spectra[:, -1] = rng.standard_normal(count) * asd_rad_per_rtHz[-1] * np.sqrt(n_samples / dt_s)
    return np.fft.irfft(spectra, n=n_samples, axis=1).astype(np.float32)


def estimate_asd_rad_per_rtHz(noise_phi: np.ndarray, dt_s: float) -> np.ndarray:
    """Estimate a one-sided ASD from independent real-valued noise records."""
    n_samples = int(noise_phi.shape[1])
    yf = np.fft.rfft(np.asarray(noise_phi, dtype=float), axis=1)
    psd = (2.0 * dt_s / n_samples) * np.mean(np.abs(yf) ** 2, axis=0)
    psd[0] *= 0.5
    if n_samples % 2 == 0:
        psd[-1] *= 0.5
    return np.sqrt(np.maximum(psd, 0.0))


def _shifted_pulse(
    transfer_fft_rad_per_eV: np.ndarray, freqs_hz: np.ndarray, dt_s: float, n_samples: int, energy_eV: float, time_s: float
) -> np.ndarray:
    phase = np.exp(-2.0j * pi * freqs_hz * float(time_s))
    return np.fft.irfft(transfer_fft_rad_per_eV * phase, n=n_samples) * (float(energy_eV) / dt_s)


def _shifted_template_pulse(
    template_fft_rad_per_eV: np.ndarray, freqs_hz: np.ndarray, n_samples: int, energy_eV: float, time_s: float
) -> np.ndarray:
    phase = np.exp(-2.0j * pi * freqs_hz * float(time_s))
    return np.fft.irfft(template_fft_rad_per_eV * phase, n=n_samples) * float(energy_eV)


def _cycled_energy_values(energies_eV: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.resize(energies_eV, count).astype(float)
    rng.shuffle(values)
    return values


def generate_pulse_sweep_dataset(
    sensor: Sensor,
    *,
    n_single: int,
    n_double: int,
    n_noise: int,
    n_energies: int,
    minimum_energy_eV: float,
    start_span_fraction: float,
    minimum_fft_frequency_hz: float,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Generate phase readout records and a template/noise optimal filter."""
    for name, count in (
        ("single sweeps", n_single),
        ("double sweeps", n_double),
        ("noise sweeps", n_noise),
    ):
        if count < 1:
            raise ValueError(f"{name} must be at least 1")
    if n_energies < 1:
        raise ValueError("number of energies must be at least 1")
    if minimum_energy_eV <= 0.0 or minimum_energy_eV > sensor.ho_decay_energy_eV:
        raise ValueError("minimum energy must be positive and no greater than the Ho energy")
    if start_span_fraction < 0.0:
        raise ValueError("start-time span fraction must be nonnegative")
    if minimum_fft_frequency_hz <= 0.0:
        raise ValueError("minimum FFT frequency must be positive")

    rng = np.random.default_rng(random_seed)
    tau_fast_s, tau_decay_s = _time_constants_s(sensor)
    dt_target_s = max(tau_fast_s / 24.0, 1.0e-9)
    filter_duration_s = 8.0 * tau_decay_s
    start_span_s = start_span_fraction * tau_decay_s
    max_lag_s = start_span_s
    pretrigger_s = 1.0 * tau_decay_s
    posttrigger_s = filter_duration_s + 1.0 * tau_decay_s
    duration_needed_s = pretrigger_s + start_span_s + max_lag_s + posttrigger_s
    record_duration_s = max(duration_needed_s, 1.0 / minimum_fft_frequency_hz)
    n_needed = int(ceil(record_duration_s / dt_target_s))
    n_samples = 1 << int(ceil(np.log2(max(n_needed, 1024))))
    if n_samples > 262144:
        raise ValueError("requested frequency range requires more than 262144 samples")
    dt_s = record_duration_s / n_samples
    t_s = np.arange(n_samples, dtype=float) * dt_s
    freqs_hz = np.fft.rfftfreq(n_samples, d=dt_s)
    reference_idx = int(ceil(pretrigger_s / dt_s))
    reference_time_s = reference_idx * dt_s

    model_asd = phase_noise_asd_rad_per_rtHz(sensor, freqs_hz)
    response_fft = phase_energy_response_fft_rad_per_eV(sensor, freqs_hz)
    noise_phi = _noise_sweeps(model_asd, n_samples, dt_s, n_noise, rng)

    ho_energy_eV = float(sensor.ho_decay_energy_eV)
    full_model_template = _shifted_pulse(response_fft, freqs_hz, dt_s, n_samples, ho_energy_eV, reference_time_s)
    filter_samples = min(int(ceil(filter_duration_s / dt_s)) + 1, n_samples - reference_idx)
    template_per_eV = full_model_template[reference_idx : reference_idx + filter_samples] / ho_energy_eV
    padded_template = np.zeros(n_samples, dtype=float)
    padded_template[:filter_samples] = template_per_eV
    template_fft = np.fft.rfft(padded_template)

    energies_eV = np.geomspace(float(minimum_energy_eV), ho_energy_eV, int(n_energies))
    single_energies_eV = _cycled_energy_values(energies_eV, n_single, rng)
    single_times_s = reference_time_s + rng.uniform(0.0, start_span_s, size=n_single)
    single_noise = _noise_sweeps(model_asd, n_samples, dt_s, n_single, rng)
    single_phi = np.asarray(
        [
            single_noise[i] + _shifted_template_pulse(template_fft, freqs_hz, n_samples, single_energies_eV[i], single_times_s[i])
            for i in range(n_single)
        ],
        dtype=np.float32,
    )

    double_energies_eV = rng.choice(energies_eV, size=(n_double, 2), replace=True)
    double_first_times_s = reference_time_s + rng.uniform(0.0, start_span_s, size=n_double)
    double_lag_s = rng.uniform(0.0, max_lag_s, size=n_double)
    double_times_s = np.column_stack((double_first_times_s, double_first_times_s + double_lag_s))
    double_noise = _noise_sweeps(model_asd, n_samples, dt_s, n_double, rng)
    double_phi = np.asarray(
        [
            double_noise[i]
            + _shifted_template_pulse(template_fft, freqs_hz, n_samples, double_energies_eV[i, 0], double_times_s[i, 0])
            + _shifted_template_pulse(template_fft, freqs_hz, n_samples, double_energies_eV[i, 1], double_times_s[i, 1])
            for i in range(n_double)
        ],
        dtype=np.float32,
    )

    estimated_asd = estimate_asd_rad_per_rtHz(noise_phi, dt_s)
    psd = np.maximum(estimated_asd**2, np.finfo(float).tiny)
    weights = np.zeros_like(psd)
    weights[1:] = 1.0 / psd[1:]
    normalization = float(np.fft.irfft(np.abs(template_fft) ** 2 * weights, n=n_samples)[0])

    return {
        "format": DATASET_FORMAT,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sensor_inputs": asdict(sensor.inputs),
        "readout": "phi",
        "units": {"time": "s", "phi": "rad", "energy": "eV", "asd": "rad/sqrt(Hz)"},
        "generation": {
            "n_samples": n_samples,
            "dt_s": dt_s,
            "tau_fast_s": tau_fast_s,
            "tau_decay_s": tau_decay_s,
            "filter_duration_s": filter_duration_s,
            "filter_samples": filter_samples,
            "record_duration_s": record_duration_s,
            "reference_time_s": reference_time_s,
            "start_span_fraction": float(start_span_fraction),
            "start_span_s": start_span_s,
            "max_double_lag_s": max_lag_s,
            "requested_minimum_fft_frequency_hz": float(minimum_fft_frequency_hz),
            "minimum_fft_frequency_hz": float(freqs_hz[1]),
            "search_start_s": reference_time_s,
            "search_stop_s": reference_time_s + start_span_s + max_lag_s,
            "random_seed": random_seed,
        },
        "time_s": t_s,
        "frequencies_hz": freqs_hz,
        "energy_grid_eV": energies_eV,
        "noise": {
            "phi": noise_phi,
            "model_asd_phi_rad_per_rtHz": model_asd,
            "estimated_asd_phi_rad_per_rtHz": estimated_asd,
        },
        "single": {
            "phi": single_phi,
            "energies_eV": single_energies_eV,
            "total_energy_eV": single_energies_eV.copy(),
            "pulse_times_s": single_times_s[:, None],
            "lag_s": np.zeros(n_single, dtype=float),
        },
        "double": {
            "phi": double_phi,
            "energies_eV": double_energies_eV,
            "total_energy_eV": np.sum(double_energies_eV, axis=1),
            "pulse_times_s": double_times_s,
            "lag_s": double_lag_s,
        },
        "optimal_filter": {
            "template_source": "model phase response",
            "template_phi_per_eV": template_per_eV,
            "template_fft_per_eV": template_fft,
            "noise_psd_phi_rad2_per_Hz": psd,
            "weights": weights,
            "normalization": normalization,
        },
    }


def _fit_one_trace(dataset: dict[str, Any], trace_phi: np.ndarray) -> tuple[float, float, float, float]:
    generation = dataset["generation"]
    filt = dataset["optimal_filter"]
    n_samples = int(generation["n_samples"])
    dt_s = float(generation["dt_s"])
    template_fft = np.asarray(filt["template_fft_per_eV"], dtype=complex)
    weights = np.asarray(filt["weights"], dtype=float)
    freqs_hz = np.asarray(dataset["frequencies_hz"], dtype=float)
    yf = np.fft.rfft(np.asarray(trace_phi, dtype=float))
    corr = np.fft.irfft(np.conjugate(template_fft) * yf * weights, n=n_samples)
    i0 = max(1, int(np.floor(float(generation["search_start_s"]) / dt_s)))
    i1 = min(n_samples - 2, int(np.ceil(float(generation["search_stop_s"]) / dt_s)) + 1)
    i_peak = i0 + int(np.argmax(corr[i0 : i1 + 1]))
    y_m, y_0, y_p = corr[i_peak - 1], corr[i_peak], corr[i_peak + 1]
    denom = y_m - 2.0 * y_0 + y_p
    frac = 0.0 if denom == 0.0 else float(np.clip(0.5 * (y_m - y_p) / denom, -0.5, 0.5))
    pulse_time_s = (i_peak + frac) * dt_s
    shift = np.exp(-2.0j * pi * freqs_hz * pulse_time_s)
    shifted_fft = template_fft * shift
    normalization = float(filt["normalization"])
    amplitude_eV = float(np.fft.irfft(np.conjugate(shifted_fft) * yf * weights, n=n_samples)[0] / normalization)
    residual_fft = yf - amplitude_eV * shifted_fft
    valid = np.arange(1, residual_fft.size - (1 if n_samples % 2 == 0 else 0))
    psd = np.asarray(filt["noise_psd_phi_rad2_per_Hz"], dtype=float)
    chi2 = float((4.0 * dt_s / n_samples) * np.sum(np.abs(residual_fft[valid]) ** 2 / psd[valid]))
    dof = max(2 * int(valid.size) - 2, 1)
    return amplitude_eV, pulse_time_s, chi2, chi2 / dof


def analyze_pulse_sweep_dataset(dataset: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Fit the single-pulse optimal filter to single and double trace categories."""
    if dataset.get("format") != DATASET_FORMAT:
        raise ValueError("not a supported pulse-sweep dataset")
    result: dict[str, dict[str, np.ndarray]] = {}
    for category in ("single", "double"):
        source = dataset[category]
        fits = np.asarray([_fit_one_trace(dataset, trace) for trace in source["phi"]], dtype=float)
        result[category] = {
            "true_total_energy_eV": np.asarray(source["total_energy_eV"], dtype=float),
            "lag_s": np.asarray(source["lag_s"], dtype=float),
            "fit_energy_eV": fits[:, 0],
            "fit_time_s": fits[:, 1],
            "chi2": fits[:, 2],
            "reduced_chi2": fits[:, 3],
        }
    return result
