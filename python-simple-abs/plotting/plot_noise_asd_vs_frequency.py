"""Interactive Tk GUI plotter for ASD/NEP noise curves using the Sensor model."""

from __future__ import annotations

from dataclasses import asdict
from math import pi
from pathlib import Path
import pickle
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensor import Sensor, Version1SensorInputs


PLOT_DIR = Path(__file__).resolve().parent
SAVES_DIR = PLOT_DIR / "saves"
SETTINGS_FILE = SAVES_DIR / "last_plot_settings.pkl"
STARTUP_STATE_FILE = SAVES_DIR / "last_plot_settings_state.pkl"

INPUT_SECTIONS = [
    (
        "Operating Point",
        [
            "T0_K",
            "Tb_K",
            "pg_drive_dBm",
            "f0_Hz",
            "detuning_widths",
            "detuning_pid_gain_Hz_per_rad",
            "detuning_pid_integrator_time_s",
            "detuning_pid_derivative_time_s",
            "detuning_pid_derivative_filter_factor",
            "amplifier_noise_temperature_K",
            "nep_sufficiency_percent",
            "event_power_fraction_kid1",
        ],
    ),
    (
        "Absorber",
        [
            "heat_capacity_eV_per_mK",
            "ho_in_au_atomic_fraction",
        ],
    ),
    (
        "TLS",
        [
            "asd_deltaC_over_C_tls_100hz_per_rtHz",
            "tls_beta",
        ],
    ),
    (
        "Resonator",
        [
            "Qi",
            "Qc",
            "tau_qp_s",
            "kinetic_inductance_fraction",
            "alpha_A",
            "alpha_phi",
            "Tc_K",
        ],
    ),
]
INPUT_KEYS = [k for _, keys in INPUT_SECTIONS for k in keys]
INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "amplifier_noise_temperature_K": ("amp_noise_temp",),
}

LABELS = {
    "T0_K": "T0 [K]",
    "Tb_K": "Tb [K]",
    "pg_drive_dBm": "Drive [dBm]",
    "f0_Hz": "f0 [Hz]",
    "detuning_widths": "Detuning [widths]",
    "detuning_pid_gain_Hz_per_rad": "PID K [Hz/rad]",
    "detuning_pid_integrator_time_s": "PID tau_i [s]",
    "detuning_pid_derivative_time_s": "PID tau_d [s]",
    "detuning_pid_derivative_filter_factor": "PID N",
    "amplifier_noise_temperature_K": "Amp Noise Temp [K]",
    "nep_sufficiency_percent": "NEP suff [%]",
    "event_power_fraction_kid1": "Event frac KID1",
    "heat_capacity_eV_per_mK": "C [eV/mK]",
    "ho_in_au_atomic_fraction": "Ho/Au frac",
    "asd_deltaC_over_C_tls_100hz_per_rtHz": "TLS dC/C ASD @100Hz",
    "tls_beta": "TLS beta",
    "Qi": "Qi",
    "Qc": "Qc",
    "tau_qp_s": "tau_qp [s]",
    "kinetic_inductance_fraction": "kinetic frac",
    "alpha_A": "alpha_A",
    "alpha_phi": "alpha_phi",
    "Tc_K": "Tc [K]",
    "feedback_heater_gain_W_per_rad": "Kp [W/rad]",
    "feedback_heater_derivative_gain_W_s_per_rad": "Kd [W*s/rad]",
}

RULE_SPECS: list[tuple[str, str]] = [
    ("Rule 1", "core_rule1_ok"),
    ("Rule 2", "core_rule2_ok"),
    ("Rule 3", "core_rule3_ok"),
    ("Rule 4", "core_rule4_ok"),
    ("Rule 5", "core_rule5_ok"),
    ("Rule 6", "core_rule6_ok"),
    ("Rule 7", "core_rule7_ok"),
    ("Rule 8", "core_rule8_ok"),
    ("Rule 9", "core_rule9_ok"),
    ("Rule 10", "core_rule10_ok"),
    ("Rule 11", "core_rule11_ok"),
    ("Rule 12", "core_rule12_ok"),
    ("Rule 13", "core_rule13_ok"),
    ("Rule 14", "core_rule14_ok"),
    ("Rule 15", "core_rule15_ok"),
]


def _positive_limits(arrays: list[np.ndarray], pad: float = 1.3) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return (1e-30, 1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return (vmin / pad, vmax * pad)


def _resolution_threshold_markers(
    f_hz: np.ndarray, nep_w_per_rthz: np.ndarray, thresholds: tuple[float, ...] = (0.50, 0.10, 0.01)
) -> list[tuple[float, str]]:
    """Return first frequencies (high->low integration) within threshold of full sigma.

    threshold=0.10 means first f where sigma(f) <= 1.10 * sigma_full.
    """
    f = np.asarray(f_hz, dtype=float)
    nep = np.asarray(nep_w_per_rthz, dtype=float)
    valid = np.isfinite(f) & np.isfinite(nep) & (f > 0.0) & (nep > 0.0)
    if np.count_nonzero(valid) < 2:
        return []
    f = f[valid]
    nep = nep[valid]
    order = np.argsort(f)
    f = f[order]
    nep = nep[order]
    integ = 4.0 / (nep * nep)
    df = np.diff(f)
    trap = 0.5 * (integ[:-1] + integ[1:]) * df
    inv_sigma2_prefix = np.zeros_like(f)
    inv_sigma2_prefix[1:] = np.cumsum(trap)
    inv_sigma2_full = float(inv_sigma2_prefix[-1])
    # Numerical guard: high->low cumulative information must be >= 0.
    inv_sigma2_cum = np.maximum(inv_sigma2_full - inv_sigma2_prefix, 0.0)
    if not np.isfinite(inv_sigma2_full) or inv_sigma2_full <= 0.0:
        return []
    sigma_full = 1.0 / np.sqrt(inv_sigma2_full)

    out: list[tuple[float, str]] = []
    for frac in thresholds:
        sigma_target = (1.0 + float(frac)) * sigma_full
        sigma_cum = np.full_like(inv_sigma2_cum, np.inf, dtype=float)
        pos = inv_sigma2_cum > 0.0
        sigma_cum[pos] = 1.0 / np.sqrt(inv_sigma2_cum[pos])
        hit = sigma_cum <= sigma_target
        idxs = np.where(hit)[0]
        if idxs.size > 0:
            out.append((float(f[idxs[-1]]), f"{int(round(100.0 * frac))}%"))
    return out


def _safe_sigma_energy_mev(s: Sensor, f_hz: np.ndarray, nep_w_per_rthz: np.ndarray) -> float:
    """Return sigma_E in meV, or NaN when a readout has no finite NEP."""
    f = np.asarray(f_hz, dtype=float)
    nep = np.asarray(nep_w_per_rthz, dtype=float)
    valid = np.isfinite(f) & np.isfinite(nep) & (f > 0.0) & (nep > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    order = np.argsort(f[valid])
    return 1.0e3 * s.sigma_energy_from_nep_spectrum_eV(f[valid][order], nep[valid][order])


class NoiseGui:
    def __init__(self) -> None:
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        self.defaults = asdict(Version1SensorInputs())
        self.defaults.setdefault("event_power_fraction_kid1", 1.0)
        (
            self.current,
            self.last_loaded,
            self.last_loaded_name,
            self.startup_ui_state,
            self.last_loaded_ui_state,
        ) = self._load_startup_settings()
        self.undo_stack: list[dict[str, float]] = []

        self.root = tk.Tk()
        self.root.title("Noise ASD / NEP Plotter")
        self.root.geometry("1500x900")

        self.mode_var = tk.StringVar(value="Noise ASD")
        self.readout_var = tk.StringVar(value="Phase")
        self.kid_power_on_var = tk.BooleanVar(value=True)
        self.feedback_on_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.loaded_name_var = tk.StringVar(value="")
        self.rule_status_vars: dict[str, tk.StringVar] = {}
        self.rule_status_labels: dict[str, ttk.Label] = {}
        self.entry_vars: dict[str, tk.StringVar] = {}
        self.entry_widgets: dict[str, ttk.Entry] = {}
        self.event_windows: list["EventResponseWindow"] = []

        self._build_layout()
        self._apply_ui_state(self.startup_ui_state)
        self._recompute_and_draw()

    @staticmethod
    def _settings_filetypes() -> list[tuple[str, str]]:
        return [("Pickle files", "*.pkl"), ("All files", "*.*")]

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=4)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

        rules_frame = ttk.Frame(self.root, padding=(10, 8, 6, 8))
        rules_frame.grid(row=0, column=0, sticky="nsw")
        rules_frame.columnconfigure(0, weight=1)
        ttk.Label(rules_frame, text="Rules", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        for i, (rule_name, _attr_name) in enumerate(RULE_SPECS, start=1):
            rule_num = rule_name.replace("Rule ", "").strip()
            row_frame = ttk.Frame(rules_frame)
            row_frame.grid(row=i, column=0, sticky="w", pady=1)
            svar = tk.StringVar(value=f"●{rule_num}")
            slbl = ttk.Label(row_frame, textvariable=svar, width=4, font=("Segoe UI", 10, "bold"))
            slbl.grid(row=0, column=0, sticky="w")
            self.rule_status_vars[rule_name] = svar
            self.rule_status_labels[rule_name] = slbl

        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        # Keep Matplotlib canvas + toolbar in a pack-managed subframe to avoid
        # mixing Tk geometry managers in the same parent.
        plot_inner = ttk.Frame(plot_frame)
        plot_inner.grid(row=0, column=0, sticky="nsew")

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_inner)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_inner)
        toolbar.update()

        controls = ttk.Frame(self.root, padding=6)
        controls.grid(row=0, column=2, sticky="ns")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Noise/NEP Controls", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(controls, textvariable=self.loaded_name_var, foreground="#222").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )

        mode_frame = ttk.LabelFrame(controls, text="Mode", padding=4)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Radiobutton(mode_frame, text="Noise ASD", variable=self.mode_var, value="Noise ASD", command=self._on_mode).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(mode_frame, text="NEP", variable=self.mode_var, value="NEP", command=self._on_mode).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )

        readout_frame = ttk.LabelFrame(controls, text="Readout", padding=4)
        readout_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Radiobutton(
            readout_frame, text="Phase", variable=self.readout_var, value="Phase", command=self._on_mode
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            readout_frame, text="Amplitude", variable=self.readout_var, value="Amplitude", command=self._on_mode
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        toggle_frame = ttk.LabelFrame(controls, text="Toggles", padding=4)
        toggle_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            toggle_frame, text="KID Power", variable=self.kid_power_on_var, command=self._on_toggle
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            toggle_frame, text="Feedback", variable=self.feedback_on_var, command=self._on_toggle
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        section_container = ttk.Frame(controls)
        section_container.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 4))
        section_container.columnconfigure(0, weight=1)

        for section_idx, (section_name, keys) in enumerate(INPUT_SECTIONS):
            section = ttk.LabelFrame(section_container, text=section_name, padding=4)
            section.grid(row=section_idx, column=0, sticky="ew", pady=(0, 4))
            split = (len(keys) + 1) // 2
            for i, key in enumerate(keys):
                block_col = 0 if i < split else 1
                r = i if i < split else i - split
                c0 = block_col * 2
                ttk.Label(section, text=LABELS[key], width=14).grid(row=r, column=c0, sticky="w", padx=(0, 4), pady=1)
                var = tk.StringVar(value=f"{self.current[key]:.8g}")
                ent = ttk.Entry(section, textvariable=var, width=11)
                ent.grid(row=r, column=c0 + 1, sticky="ew", pady=1, padx=(0, 6))
                ent.bind("<Return>", self._on_field_commit)
                ent.bind("<FocusOut>", self._on_field_commit)
                self.entry_vars[key] = var
                self.entry_widgets[key] = ent
            section.columnconfigure(1, weight=1)
            section.columnconfigure(3, weight=1)

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(1, 3))

        ttk.Button(button_frame, text="Defaults", command=self._load_defaults).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Undo", command=self._undo).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Save", command=self._save_current).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Load", command=self._load_saved).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Restore", command=self._restore_last_loaded).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Event Response", command=self._open_event_response).grid(
            row=2, column=0, columnspan=3, padx=2, pady=2, sticky="ew"
        )
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        ttk.Label(controls, textvariable=self.status_var, wraplength=300, foreground="#444").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )

        ttk.Label(
            controls,
            textvariable=self.summary_var,
            justify="right",
            anchor="e",
            wraplength=320,
            foreground="#222",
        ).grid(row=8, column=0, columnspan=2, sticky="e", pady=(6, 0))

    @staticmethod
    def _loaded_value_with_aliases(loaded_settings: dict, key: str) -> float | None:
        if key in loaded_settings:
            return float(loaded_settings[key])
        for alias in INPUT_ALIASES.get(key, ()):
            if alias in loaded_settings:
                return float(loaded_settings[alias])
        return None

    def _load_saved_or_defaults(self) -> dict[str, float]:
        if SETTINGS_FILE.exists():
            try:
                with SETTINGS_FILE.open("rb") as f:
                    loaded = pickle.load(f)
                loaded_settings = self._extract_settings_dict(loaded)
                if isinstance(loaded_settings, dict):
                    merged = dict(self.defaults)
                    for k in INPUT_KEYS:
                        v = self._loaded_value_with_aliases(loaded_settings, k)
                        if v is not None:
                            merged[k] = v
                    return merged
            except Exception:
                pass
        return dict(self.defaults)

    def _load_settings_file(self, path: Path) -> dict[str, float] | None:
        try:
            with path.open("rb") as f:
                loaded = pickle.load(f)
            loaded_settings = self._extract_settings_dict(loaded)
            if not isinstance(loaded_settings, dict):
                return None
            vals = dict(self.defaults)
            for k in INPUT_KEYS:
                v = self._loaded_value_with_aliases(loaded_settings, k)
                if v is not None:
                    vals[k] = v
            return vals
        except Exception:
            return None

    @staticmethod
    def _extract_settings_dict(loaded: object) -> dict | None:
        if not isinstance(loaded, dict):
            return None
        if "settings" in loaded and isinstance(loaded["settings"], dict):
            return loaded["settings"]
        return loaded

    @staticmethod
    def _extract_ui_state(loaded: object) -> dict[str, str]:
        if isinstance(loaded, dict) and isinstance(loaded.get("ui"), dict):
            ui = loaded["ui"]
            out: dict[str, str] = {}
            for k in ("mode", "readout", "kid_power_on", "feedback_on"):
                v = ui.get(k)
                if isinstance(v, str):
                    out[k] = v
            return out
        return {}

    def _current_ui_state(self) -> dict[str, str]:
        return {
            "mode": self.mode_var.get(),
            "readout": self.readout_var.get(),
            "kid_power_on": "on" if bool(self.kid_power_on_var.get()) else "off",
            "feedback_on": "on" if bool(self.feedback_on_var.get()) else "off",
        }

    def _apply_ui_state(self, ui: dict[str, str]) -> None:
        mode = ui.get("mode")
        if mode in ("Noise ASD", "NEP"):
            self.mode_var.set(mode)
        readout = ui.get("readout")
        if readout in ("Phase", "Amplitude"):
            self.readout_var.set(readout)
        kid_power_on = ui.get("kid_power_on")
        if kid_power_on in ("on", "off"):
            self.kid_power_on_var.set(kid_power_on == "on")
        feedback_on = ui.get("feedback_on")
        if feedback_on in ("on", "off"):
            self.feedback_on_var.set(feedback_on == "on")

    def _load_startup_settings(self) -> tuple[dict[str, float], dict[str, float], str | None, dict[str, str], dict[str, str]]:
        # Startup should always reflect current code defaults.
        # Saved presets remain available via explicit Load/Restore actions.
        vals = dict(self.defaults)
        return vals, dict(vals), None, {}, {}

    def _persist_startup_state(self, source_path: Path) -> None:
        try:
            with STARTUP_STATE_FILE.open("wb") as f:
                pickle.dump(
                    {
                        "source_path": str(source_path),
                        "ui": self._current_ui_state(),
                    },
                    f,
                )
        except Exception:
            pass

    def _read_fields(self) -> dict[str, float]:
        vals = dict(self.current)
        for k, var in self.entry_vars.items():
            vals[k] = float(var.get().strip())
        return vals

    def _write_fields(self, vals: dict[str, float]) -> None:
        for k, var in self.entry_vars.items():
            var.set(f"{vals[k]:.8g}")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _set_summary(self, s: Sensor) -> None:
        delta_t_mk = 1.0e3 * float(s.deltaT_event_full_absorption_K)
        rate_hz = float(s.count_rate_Hz)
        pileup_pct = 100.0 * float(s.pileup_probability_max)
        shorten = float(s.mt_pulse_shortening_ratio)
        eigs = np.array(s.mt_eigenvalues, dtype=complex)
        pulse_tau_s = float("nan")
        if s.mt_stable:
            slowest_decay_per_s = float(np.min(-np.real(eigs)))
            if slowest_decay_per_s > 0.0:
                pulse_tau_s = 1.0 / slowest_decay_per_s
        pulse_tau_txt = f"{pulse_tau_s:.3g} s" if np.isfinite(pulse_tau_s) else "n/a"
        self.summary_var.set(
            "KID power: "
            f"{'On' if self.kid_power_on_var.get() else 'Off'}\n"
            "Feedback: "
            f"{'On' if self.feedback_on_var.get() else 'Off'}\n"
            "Event dT (island): "
            f"{delta_t_mk:.3g} mK\n"
            "Average event rate: "
            f"{rate_hz:.3g} Hz\n"
            "Pileup probability: "
            f"{pileup_pct:.3g}%\n"
            "Pulse shortening factor: "
            f"{shorten:.3g}\n"
            "Event pulse time: "
            f"{pulse_tau_txt}\n"
            "C absorber: "
            f"{float(s.C_eV_per_mK):.3g} eV/mK\n"
            "G thermal link: "
            f"{float(s.G_W_per_K):.3e} W/K\n"
            "tau_th = C/G: "
            f"{float(s.tau_th_s):.3g} s"
        )

    @staticmethod
    def _settings_match(a: dict[str, float], b: dict[str, float], rtol: float = 1.0e-12, atol: float = 0.0) -> bool:
        return all(np.isclose(float(a[k]), float(b[k]), rtol=rtol, atol=atol) for k in INPUT_KEYS)

    def _update_loaded_name(self) -> None:
        if self.last_loaded_name and self._settings_match(self.current, self.last_loaded):
            self.loaded_name_var.set(f"Settings file: {self.last_loaded_name}")
        else:
            self.loaded_name_var.set("")

    def _push_undo(self, prev: dict[str, float]) -> None:
        self.undo_stack.append(dict(prev))
        if len(self.undo_stack) > 20:
            self.undo_stack = self.undo_stack[-20:]

    def _build_sensor(self, settings: dict[str, float]) -> Sensor:
        kwargs = dict(self.defaults)
        kwargs.update(settings)
        # UI toggles are runtime overrides; keep entry values unchanged.
        if not bool(self.kid_power_on_var.get()):
            kwargs["pg_drive_dBm"] = -300.0
        if (not bool(self.feedback_on_var.get())) or (not bool(self.kid_power_on_var.get())):
            kwargs["detuning_pid_gain_Hz_per_rad"] = 0.0
        ctor_keys = set(asdict(Version1SensorInputs()).keys())
        clean = {k: v for k, v in kwargs.items() if k in ctor_keys}
        clean["event_power_fraction_kid1"] = 1.0
        return Sensor(Version1SensorInputs(**clean))

    def _compute_data(self, s: Sensor) -> dict[str, object]:
        eigs = np.array(s.mt_eigenvalues, dtype=complex)
        max_mode_rate_per_s = float(np.max(np.abs(eigs)))
        f_min_hz = 0.1
        f_max_hz = max(f_min_hz * 10.0, 10.0 * max_mode_rate_per_s / (2.0 * pi))
        freqs_hz = np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), 1000)

        asd_phase_johnson = np.zeros_like(freqs_hz)
        asd_phase_phonon = np.zeros_like(freqs_hz)
        asd_phase_tls = np.zeros_like(freqs_hz)
        asd_phase_electronic = np.zeros_like(freqs_hz)
        asd_phase_amplifier = np.zeros_like(freqs_hz)
        asd_amp_johnson = np.zeros_like(freqs_hz)
        asd_amp_phonon = np.zeros_like(freqs_hz)
        asd_amp_tls = np.zeros_like(freqs_hz)
        asd_amp_electronic = np.zeros_like(freqs_hz)
        asd_amp_amplifier = np.zeros_like(freqs_hz)
        phase_resp = np.zeros_like(freqs_hz)
        amp_resp = np.zeros_like(freqs_hz)

        for i, f_hz in enumerate(freqs_hz):
            y_j_a1 = s._propagate_noise_vector(s.n_johnson_A_1(), f_hz)
            y_j_phi1 = s._propagate_noise_vector(s.n_johnson_phi_1(), f_hz)
            y_ph1 = s._propagate_noise_vector(s.n_phonon_1(), f_hz)
            y_tls = s._propagate_noise_vector(s.n_tls_phi_at_hz(float(f_hz)), f_hz)
            y_e_a1 = s._propagate_noise_vector(s.n_electronic_A_1(), f_hz)
            y_e_phi1 = s._propagate_noise_vector(s.n_electronic_phi_1(), f_hz)
            y_amp_a = s.y_amplifier_A_at_hz(float(f_hz))
            y_amp_phi = s.y_amplifier_phi_at_hz(float(f_hz))

            asd_phase_johnson[i] = np.sqrt(abs(y_j_a1[1]) ** 2 + abs(y_j_phi1[1]) ** 2)
            asd_phase_phonon[i] = abs(y_ph1[1])
            asd_phase_tls[i] = abs(y_tls[1])
            asd_phase_electronic[i] = np.sqrt(abs(y_e_a1[1]) ** 2 + abs(y_e_phi1[1]) ** 2)
            asd_phase_amplifier[i] = np.sqrt(abs(y_amp_a[1]) ** 2 + abs(y_amp_phi[1]) ** 2)
            asd_amp_johnson[i] = np.sqrt(abs(y_j_a1[0]) ** 2 + abs(y_j_phi1[0]) ** 2)
            asd_amp_phonon[i] = abs(y_ph1[0])
            asd_amp_tls[i] = abs(y_tls[0])
            asd_amp_electronic[i] = np.sqrt(abs(y_e_a1[0]) ** 2 + abs(y_e_phi1[0]) ** 2)
            asd_amp_amplifier[i] = np.sqrt(abs(y_amp_a[0]) ** 2 + abs(y_amp_phi[0]) ** 2)
            phase_resp[i] = s.phase_responsivity_mag_rad_per_W_at_hz(float(f_hz))
            y_unit_power = s._solve_response_vector(
                (0.0 + 0.0j, 0.0 + 0.0j, s.event_power_fraction_kid1_clamped + 0.0j),
                float(f_hz),
            )
            amp_resp[i] = abs(y_unit_power[0])

        asd_phase_total = np.sqrt(asd_phase_johnson**2 + asd_phase_phonon**2 + asd_phase_tls**2 + asd_phase_electronic**2 + asd_phase_amplifier**2)
        asd_amp_total = np.sqrt(asd_amp_johnson**2 + asd_amp_phonon**2 + asd_amp_tls**2 + asd_amp_electronic**2 + asd_amp_amplifier**2)
        asd_tls_direct = np.array([s.tls_phi_asd_at_hz_per_rtHz(float(f)) for f in freqs_hz], dtype=float)
        asd_johnson_simple = abs(s.f0_Hz * s.dphi_df_detuning_per_hz) * np.sqrt(s.sf_over_f0sq_johnson_simple)
        nep_phase_johnson = np.where(phase_resp > 0.0, asd_phase_johnson / phase_resp, np.nan)
        nep_phase_phonon = np.where(phase_resp > 0.0, asd_phase_phonon / phase_resp, np.nan)
        nep_phase_tls = np.where(phase_resp > 0.0, asd_phase_tls / phase_resp, np.nan)
        nep_phase_electronic = np.where(phase_resp > 0.0, asd_phase_electronic / phase_resp, np.nan)
        nep_phase_amplifier = np.where(phase_resp > 0.0, asd_phase_amplifier / phase_resp, np.nan)
        nep_phase_total = np.where(phase_resp > 0.0, asd_phase_total / phase_resp, np.nan)
        nep_amp_johnson = np.where(amp_resp > 0.0, asd_amp_johnson / amp_resp, np.nan)
        nep_amp_phonon = np.where(amp_resp > 0.0, asd_amp_phonon / amp_resp, np.nan)
        nep_amp_tls = np.where(amp_resp > 0.0, asd_amp_tls / amp_resp, np.nan)
        nep_amp_electronic = np.where(amp_resp > 0.0, asd_amp_electronic / amp_resp, np.nan)
        nep_amp_amplifier = np.where(amp_resp > 0.0, asd_amp_amplifier / amp_resp, np.nan)
        nep_amp_total = np.where(amp_resp > 0.0, asd_amp_total / amp_resp, np.nan)

        sigma_e_phase_mev = _safe_sigma_energy_mev(s, freqs_hz, nep_phase_total)
        sigma_e_amp_mev = _safe_sigma_energy_mev(s, freqs_hz, nep_amp_total)

        marker_specs = [
            (float(s.count_rate_Hz), "f_rate", ":"),
            (1.0 / (2.0 * pi * s.tau_th_s), "f_therm", "--"),
            (1.0 / (2.0 * pi * s.tau_res_s), "f_res", "--"),
        ]
        marker_specs.extend((abs(complex(lam)) / (2.0 * pi), f"f_eig{i + 1}", "--") for i, lam in enumerate(eigs))
        valid_markers = [
            (float(fm), name, linestyle) for fm, name, linestyle in marker_specs if np.isfinite(fm) and fm > 0.0
        ]
        valid_markers.sort(key=lambda x: x[0])

        asd_phase_ylim = _positive_limits(
            [asd_phase_johnson, asd_phase_phonon, asd_phase_tls, asd_phase_electronic, asd_phase_amplifier, asd_phase_total, asd_tls_direct, np.array([asd_johnson_simple])]
        )
        asd_amp_ylim = _positive_limits([asd_amp_johnson, asd_amp_phonon, asd_amp_tls, asd_amp_electronic, asd_amp_amplifier, asd_amp_total])
        nep_phase_ylim = _positive_limits([nep_phase_johnson, nep_phase_phonon, nep_phase_tls, nep_phase_electronic, nep_phase_amplifier, nep_phase_total])
        nep_amp_ylim = _positive_limits([nep_amp_johnson, nep_amp_phonon, nep_amp_tls, nep_amp_electronic, nep_amp_amplifier, nep_amp_total])

        return {
            "sensor": s,
            "freqs": freqs_hz,
            "asd_phase": (asd_phase_johnson, asd_phase_phonon, asd_phase_tls, asd_phase_electronic, asd_phase_amplifier, asd_phase_total),
            "asd_amp": (asd_amp_johnson, asd_amp_phonon, asd_amp_tls, asd_amp_electronic, asd_amp_amplifier, asd_amp_total),
            "asd_tls_direct": asd_tls_direct,
            "nep_phase": (nep_phase_johnson, nep_phase_phonon, nep_phase_tls, nep_phase_electronic, nep_phase_amplifier, nep_phase_total),
            "nep_amp": (nep_amp_johnson, nep_amp_phonon, nep_amp_tls, nep_amp_electronic, nep_amp_amplifier, nep_amp_total),
            "res_threshold_phase": _resolution_threshold_markers(freqs_hz, nep_phase_total),
            "res_threshold_amp": _resolution_threshold_markers(freqs_hz, nep_amp_total),
            "asd_johnson_simple": asd_johnson_simple,
            "sigma_phase_mev": sigma_e_phase_mev,
            "sigma_amp_mev": sigma_e_amp_mev,
            "markers": valid_markers,
            "asd_phase_ylim": asd_phase_ylim,
            "asd_amp_ylim": asd_amp_ylim,
            "nep_phase_ylim": nep_phase_ylim,
            "nep_amp_ylim": nep_amp_ylim,
        }

    def _draw(self) -> None:
        d = self.data
        s: Sensor = d["sensor"]  # type: ignore[assignment]
        freqs = d["freqs"]
        readout = self.readout_var.get()
        is_phase = readout == "Phase"
        asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_amplifier, asd_total = d["asd_phase"] if is_phase else d["asd_amp"]
        nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_amplifier, nep_total = d["nep_phase"] if is_phase else d["nep_amp"]

        self.ax.clear()
        mode = self.mode_var.get()
        if mode == "NEP":
            ysets = (nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_amplifier, nep_total)
            ylab = "NEP [W/rtHz]"
            title = f"Noise-Equivalent Power vs Frequency ({readout} readout)"
            self.ax.set_ylim(*(d["nep_phase_ylim"] if is_phase else d["nep_amp_ylim"]))
        else:
            ysets = (asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_amplifier, asd_total)
            ylab = "Phase ASD [rad/rtHz]" if is_phase else "Amplitude ASD [1/rtHz]"
            title = f"Noise ASD vs Frequency ({readout} readout)"
            self.ax.set_ylim(*(d["asd_phase_ylim"] if is_phase else d["asd_amp_ylim"]))

        self.ax.loglog(freqs, ysets[0], label="Johnson", color="tab:blue")
        self.ax.loglog(freqs, ysets[1], label="Phonon", color="tab:orange")
        self.ax.loglog(freqs, ysets[2], label="TLS", color="tab:green")
        self.ax.loglog(freqs, ysets[3], label="Electronic", color="tab:red")
        self.ax.loglog(freqs, ysets[4], label="Amplifier", color="tab:purple")
        self.ax.loglog(freqs, ysets[5], color="k", linestyle=":", linewidth=2.2, label="Total (quadrature)")

        if mode == "Noise ASD" and is_phase:
            self.ax.axhline(
                d["asd_johnson_simple"],
                linestyle=":",
                linewidth=1.8,
                color="tab:blue",
                alpha=0.9,
                label="Johnson (simple estimate)",
            )
            self.ax.loglog(
                freqs,
                d["asd_tls_direct"],
                linestyle=":",
                linewidth=2.0,
                color="tab:green",
                alpha=0.9,
                label="TLS (direct beta law)",
            )
        if mode == "NEP":
            phonon_ref = np.asarray(nep_phonon, dtype=float)
            phonon_ref = phonon_ref[np.isfinite(phonon_ref) & (phonon_ref > 0.0)]
            if phonon_ref.size > 0:
                self.ax.axhline(
                    float(phonon_ref[0]),
                    linestyle=":",
                    linewidth=1.8,
                    color="tab:orange",
                    alpha=0.9,
                    label="Phonon (low-f ref)",
                )

        y_levels = [0.96, 0.82, 0.68, 0.54, 0.40, 0.26]
        cluster_step_log10 = 0.035
        prev_log10_f = None
        cluster_idx = -1
        for f_mark_hz, name, linestyle in d["markers"]:
            self.ax.axvline(f_mark_hz, linestyle=linestyle, linewidth=1.0, alpha=0.7, color="#666")
            log10_f = float(np.log10(f_mark_hz))
            if prev_log10_f is None or (log10_f - prev_log10_f) > cluster_step_log10:
                cluster_idx = 0
            else:
                cluster_idx += 1
            y_text = y_levels[cluster_idx % len(y_levels)]
            self.ax.text(f_mark_hz, y_text, name, rotation=90, va="top", ha="right", transform=self.ax.get_xaxis_transform())
            prev_log10_f = log10_f

        threshold_marks = d["res_threshold_phase"] if is_phase else d["res_threshold_amp"]
        for f_thr_hz, label in threshold_marks:
            idx = int(np.argmin(np.abs(freqs - f_thr_hz)))
            y_thr = float(ysets[5][idx])
            if np.isfinite(y_thr) and y_thr > 0.0:
                self.ax.plot([f_thr_hz], [y_thr], linestyle="None", marker="|", markersize=11, color="k", zorder=5)
                self.ax.text(f_thr_hz, y_thr * 1.18, label, ha="center", va="bottom", color="k")

        if s.mt_stable:
            sigma_mev = d["sigma_phase_mev"] if is_phase else d["sigma_amp_mev"]
            if np.isfinite(sigma_mev):
                label = f"Estimated energy resolution: sigma_E = {sigma_mev:.3f} meV"
                color = "black"
            else:
                label = f"{readout} NEP unavailable: zero or invalid signal responsivity"
                color = "red"
        else:
            label = "UNSTABLE: Mt eigenvalues indicate instability"
            color = "red"
        self.ax.text(
            0.98,
            0.02,
            label,
            color=color,
            transform=self.ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        eigs = np.array(s.mt_eigenvalues, dtype=complex)
        is_underdamped = bool(np.any(np.abs(np.imag(eigs)) > 1.0e-12))
        show_underdamped = is_underdamped and bool(s.mt_stable)
        if show_underdamped:
            self.ax.text(
                0.98,
                0.075,
                "Underdamped (complex Mt eigenvalue present)",
                color="#d17a00",
                transform=self.ax.transAxes,
                ha="right",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )

        self.ax.set_title(title)
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel(ylab)
        self.ax.grid(True, which="both", alpha=0.25)
        legend_loc = "lower left" if mode == "Noise ASD" else "upper left"
        self.ax.legend(loc=legend_loc, framealpha=0.9)
        self.canvas.draw_idle()

    def _recompute_and_draw(self) -> None:
        sensor = self._build_sensor(self.current)
        self.data = self._compute_data(sensor)
        self._set_summary(sensor)
        self._update_rule_indicators(sensor)
        self._draw()
        self._sync_event_windows(sensor)

    def _update_rule_indicators(self, s: Sensor) -> None:
        style = ttk.Style(self.root)
        style.configure("RulePass.TLabel", foreground="#198754")
        style.configure("RuleFail.TLabel", foreground="#c62828")
        style.configure("RuleUnknown.TLabel", foreground="#6c757d")
        for rule_name, attr_name in RULE_SPECS:
            var = self.rule_status_vars.get(rule_name)
            lbl = self.rule_status_labels.get(rule_name)
            if var is None or lbl is None:
                continue
            try:
                raw = getattr(s, attr_name)
                passed = bool(raw)
                rule_num = rule_name.replace("Rule ", "").strip()
                var.set(f"●{rule_num}")
                lbl.configure(style="RulePass.TLabel" if passed else "RuleFail.TLabel")
            except Exception:
                rule_num = rule_name.replace("Rule ", "").strip()
                var.set(f"●{rule_num}")
                lbl.configure(style="RuleUnknown.TLabel")

    def _on_mode(self) -> None:
        self._draw()

    def _on_toggle(self) -> None:
        self._recompute_and_draw()

    def _apply_from_fields(self) -> None:
        try:
            new_vals = self._read_fields()
            prev = dict(self.current)
            self.current = new_vals
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status("Applied settings")
        except Exception as exc:
            self._set_status(f"Apply failed: {exc}")

    def _on_field_commit(self, _event=None) -> None:
        self._apply_from_fields()

    def _load_defaults(self) -> None:
        prev = dict(self.current)
        self.current = dict(self.defaults)
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status("Loaded defaults")
        except Exception as exc:
            self.current = prev
            self._write_fields(self.current)
            self._set_status(f"Defaults failed: {exc}")

    def _save_current(self) -> None:
        try:
            self.current = self._read_fields()
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save Plot Settings",
                initialdir=str(SETTINGS_FILE.parent),
                initialfile=SETTINGS_FILE.name,
                defaultextension=".pkl",
                filetypes=self._settings_filetypes(),
            )
            if not path:
                self._set_status("Save cancelled")
                return
            save_path = Path(path)
            save_settings = {k: self.current[k] for k in INPUT_KEYS}
            save_settings["amp_noise_temp"] = self.current["amplifier_noise_temperature_K"]
            with save_path.open("wb") as f:
                pickle.dump(
                    {
                        "settings": save_settings,
                        "ui": self._current_ui_state(),
                    },
                    f,
                )
            self.last_loaded = dict(self.current)
            self.last_loaded_name = save_path.name
            self.last_loaded_ui_state = self._current_ui_state()
            self._persist_startup_state(save_path)
            self._update_loaded_name()
            self._set_status(f"Saved settings to {save_path.name}")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")

    def _load_saved(self) -> None:
        try:
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Load Plot Settings",
                initialdir=str(SETTINGS_FILE.parent),
                filetypes=self._settings_filetypes(),
            )
            if not path:
                self._set_status("Load cancelled")
                return
            load_path = Path(path)
            with load_path.open("rb") as f:
                loaded = pickle.load(f)
            vals = self._load_settings_file(load_path)
            if vals is None:
                raise ValueError("invalid settings file")
            prev = dict(self.current)
            self.current = vals
            self.last_loaded = dict(vals)
            self.last_loaded_name = load_path.name
            self.last_loaded_ui_state = self._extract_ui_state(loaded)
            self._persist_startup_state(load_path)
            self._write_fields(self.current)
            self._apply_ui_state(self.last_loaded_ui_state)
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status(f"Loaded settings from {load_path.name}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    def _restore_last_loaded(self) -> None:
        prev = dict(self.current)
        prev_ui = self._current_ui_state()
        self.current = dict(self.last_loaded)
        self._write_fields(self.current)
        try:
            self._apply_ui_state(self.last_loaded_ui_state)
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status("Restored last loaded settings")
        except Exception as exc:
            self.current = prev
            self._write_fields(self.current)
            self._apply_ui_state(prev_ui)
            self._set_status(f"Restore failed: {exc}")

    def _undo(self) -> None:
        if not self.undo_stack:
            self._set_status("Undo stack empty")
            return
        prev = self.undo_stack.pop()
        self.current = prev
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._update_loaded_name()
            self._set_status("Undo applied")
        except Exception as exc:
            self._set_status(f"Undo failed: {exc}")

    def _open_event_response(self) -> None:
        try:
            self.current = self._read_fields()
            sensor = self._build_sensor(self.current)
            win = EventResponseWindow(self.root, sensor, on_close=self._on_event_window_close)
            self.event_windows.append(win)
        except Exception as exc:
            self._set_status(f"Event response failed: {exc}")

    def _sync_event_windows(self, sensor: Sensor) -> None:
        alive: list["EventResponseWindow"] = []
        for win in self.event_windows:
            if win.is_closed:
                continue
            try:
                win.update_sensor(sensor)
                alive.append(win)
            except Exception:
                continue
        self.event_windows = alive

    def _on_event_window_close(self, win: "EventResponseWindow") -> None:
        self.event_windows = [w for w in self.event_windows if (w is not win and not w.is_closed)]

    def run(self) -> None:
        startup = self.last_loaded_name if self.last_loaded_name else "defaults"
        self._update_loaded_name()
        self._set_status(f"Loaded startup settings ({startup})")
        self.root.mainloop()


def main() -> None:
    app = NoiseGui()
    app.run()


class EventResponseWindow:
    def __init__(self, parent: tk.Tk, sensor: Sensor, on_close=None) -> None:
        self.sensor = sensor
        self.on_close = on_close
        self.is_closed = False
        self.win = tk.Toplevel(parent)
        self.win.title("Ho Event Response")
        self.win.geometry("1200x760")
        self.win.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.domain_var = tk.StringVar(value="Time")
        self.time_axis_var = tk.StringVar(value="Linear")
        self.var_var = tk.StringVar(value="phi")
        self._build()
        self._recompute_and_draw()

    def _build(self) -> None:
        self.win.columnconfigure(0, weight=1)
        self.win.columnconfigure(1, weight=0)
        self.win.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.win, padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        ctl = ttk.Frame(self.win, padding=8)
        ctl.grid(row=0, column=1, sticky="ns")
        ttk.Label(ctl, text="Event Response", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        domain = ttk.LabelFrame(ctl, text="Domain", padding=4)
        domain.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(domain, text="Time", variable=self.domain_var, value="Time", command=self._draw).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(domain, text="Frequency", variable=self.domain_var, value="Frequency", command=self._draw).grid(row=0, column=1, sticky="w", padx=(8, 0))

        taxis = ttk.LabelFrame(ctl, text="Time Axis", padding=4)
        taxis.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(taxis, text="Linear", variable=self.time_axis_var, value="Linear", command=self._draw).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(taxis, text="Log", variable=self.time_axis_var, value="Log", command=self._draw).grid(row=0, column=1, sticky="w", padx=(8, 0))

        varf = ttk.LabelFrame(ctl, text="Variable", padding=4)
        varf.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        names = [("r", "r"), ("phi", "phi"), ("T1", "T1"), ("dx", "dx")]
        for i, (lbl, v) in enumerate(names):
            ttk.Radiobutton(varf, text=lbl, variable=self.var_var, value=v, command=self._draw).grid(row=i, column=0, sticky="w")

        self.status = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.status, wraplength=260, foreground="#333").grid(row=4, column=0, sticky="w")

    @staticmethod
    def _var_index(var: str) -> int:
        return {"r": 0, "phi": 1, "T1": 2, "dx": 3}[var]

    @staticmethod
    def _var_units(var: str) -> tuple[str, str]:
        # Response units are from event-energy forcing:
        # r -> dimensionless, phi -> rad, T1/T2 -> K.
        if var == "r":
            return ("1", "1·s")
        if var == "phi":
            return ("rad", "rad*s")
        if var == "dx":
            return ("1", "1*s")
        return ("K", "K*s")


    def _var_time_series(self, var: str) -> np.ndarray:
        if var in ("r", "phi", "T1", "dx"):
            idx = self._var_index(var)
            if idx >= self.y_t.shape[0]:
                return np.zeros_like(self.t_s, dtype=complex)
            return self.y_t[idx]
        return np.zeros_like(self.t_s, dtype=complex)

    def _recompute_and_draw(self) -> None:
        s = self.sensor
        mt = np.array(s.mt_matrix, dtype=complex)
        d1 = np.array(s.d1_matrix, dtype=complex)
        src = np.array((0.0 + 0.0j, 0.0 + 0.0j, s.event_power_fraction_kid1_clamped + 0.0j), dtype=complex)
        b = np.linalg.solve(d1, src)

        evals, evecs = np.linalg.eig(mt)
        vinv = np.linalg.inv(evecs)
        neg = np.real(evals) < 0.0
        t_const = -1.0 / np.real(evals[neg]) if np.any(neg) else np.array([1.0])
        tau_min = float(np.min(t_const))
        tau_max = float(np.max(t_const))
        t_max = max(8.0 * tau_max, 1.0e-5)
        # Adaptive uniform sampling: ensure we resolve fast modes while covering slow modes.
        # Use ~24 points on the fastest time constant, then round to power-of-two for FFT.
        dt_target = max(tau_min / 24.0, 1.0e-9)
        n_needed = int(np.ceil(t_max / dt_target)) + 1
        n_pow2 = 1 << int(np.ceil(np.log2(max(n_needed, 1024))))
        n_samples = min(max(n_pow2, 1024), 262144)
        self.t_s = np.linspace(0.0, t_max, n_samples)
        self._time_grid_info = {
            "tau_min_s": tau_min,
            "tau_max_s": tau_max,
            "n_needed": n_needed,
            "n_used": n_samples,
            "clipped": n_samples < n_needed,
        }

        coeff = vinv @ b
        exp_terms = np.exp(np.outer(evals, self.t_s))
        h = evecs @ (coeff[:, None] * exp_terms)
        y3_t = h * s.ho_decay_energy_J

        dt = float(self.t_s[1] - self.t_s[0])
        f_all = np.fft.fftfreq(self.t_s.size, d=dt)
        hf = np.zeros((4, self.t_s.size), dtype=complex)
        src3 = tuple(complex(v) for v in src[:3])
        for i, f_hz in enumerate(f_all):
            hf[:, i] = s._solve_response_vector(src3, float(f_hz)) * s.ho_decay_energy_J
        dx_t = np.fft.ifft(hf[3] / float(self.t_s[1] - self.t_s[0]))
        self.y_t = np.vstack((y3_t, dx_t))

        pos = f_all >= 0.0
        freqs = f_all[pos]
        y_fft_all = np.fft.fft(self.y_t, axis=1) * dt
        self.freq_hz = freqs
        self.y_fft = y_fft_all[:, pos]

        # Matrix frequency response for unit power input.
        h_f = np.zeros((self.y_t.shape[0], freqs.size), dtype=complex)
        for i, f_hz in enumerate(freqs):
            h_f[:, i] = s._solve_response_vector(src3, float(f_hz))
        self.h_f_matrix = h_f * s.ho_decay_energy_J
        self._draw()

    def update_sensor(self, sensor: Sensor) -> None:
        if self.is_closed:
            return
        self.sensor = sensor
        self._recompute_and_draw()

    def _handle_close(self) -> None:
        self.is_closed = True
        try:
            if callable(self.on_close):
                self.on_close(self)
        finally:
            self.win.destroy()

    def _draw(self) -> None:
        var_name = self.var_var.get()
        idx = self._var_index(var_name) if var_name in ("r", "phi", "T1", "dx") else -1
        if idx >= 0 and idx >= self.y_t.shape[0]:
            self.var_var.set("phi")
            var_name = "phi"
            idx = self._var_index(var_name)
        unit_t, unit_f = self._var_units(var_name)
        y_series = self._var_time_series(var_name)
        self.ax.clear()
        if self.domain_var.get() == "Time":
            y = np.real(y_series)
            t_ms = self.t_s * 1e3
            if self.time_axis_var.get() == "Log":
                pos = t_ms > 0.0
                if var_name == "L_terms":
                    y1 = np.real(self._var_time_series("L1_term"))
                    y2 = np.real(self._var_time_series("L2_term"))
                    r = np.real(self._var_time_series("r"))
                    rdot = np.gradient(r, self.t_s)
                    rterm = ((4.0 * float(self.sensor.Qi_eff) * float(self.sensor.Qr) * float(self.sensor.x)) / (2.0 * pi * float(self.sensor.f0_Hz))) * rdot - float(self.sensor.beta_phi) * r
                    ysum = y1 + y2
                    self.ax.semilogx(t_ms[pos], y1[pos], color="tab:blue", label="L1 term")
                    self.ax.semilogx(t_ms[pos], y2[pos], color="tab:orange", label="L2 term")
                    self.ax.semilogx(t_ms[pos], ysum[pos], color="black", linestyle=":", label="L1+L2")
                    self.ax.semilogx(t_ms[pos], rterm[pos], color="tab:green", label="r term")
                    self.ax.legend(loc="best")
                else:
                    self.ax.semilogx(t_ms[pos], y[pos], color="tab:blue")
                self.ax.set_xlabel("Time [ms] (log)")
            else:
                if var_name == "L_terms":
                    y1 = np.real(self._var_time_series("L1_term"))
                    y2 = np.real(self._var_time_series("L2_term"))
                    r = np.real(self._var_time_series("r"))
                    rdot = np.gradient(r, self.t_s)
                    rterm = ((4.0 * float(self.sensor.Qi_eff) * float(self.sensor.Qr) * float(self.sensor.x)) / (2.0 * pi * float(self.sensor.f0_Hz))) * rdot - float(self.sensor.beta_phi) * r
                    ysum = y1 + y2
                    self.ax.plot(t_ms, y1, color="tab:blue", label="L1 term")
                    self.ax.plot(t_ms, y2, color="tab:orange", label="L2 term")
                    self.ax.plot(t_ms, ysum, color="black", linestyle=":", label="L1+L2")
                    self.ax.plot(t_ms, rterm, color="tab:green", label="r term")
                    self.ax.legend(loc="best")
                else:
                    self.ax.plot(t_ms, y, color="tab:blue")
                self.ax.set_xlabel("Time [ms]")
            self.ax.set_ylabel(f"{var_name}(t) [{unit_t}]")
            self.ax.set_title(f"Ho Event Response: {var_name}(t)")
        else:
            dt = float(self.t_s[1] - self.t_s[0])
            if var_name == "L_terms":
                y1 = self._var_time_series("L1_term")
                y2 = self._var_time_series("L2_term")
                r = np.real(self._var_time_series("r"))
                rdot = np.gradient(r, self.t_s)
                rterm = ((4.0 * float(self.sensor.Qi_eff) * float(self.sensor.Qr) * float(self.sensor.x)) / (2.0 * pi * float(self.sensor.f0_Hz))) * rdot - float(self.sensor.beta_phi) * r
                ysum = y1 + y2
                yf1 = np.abs(np.fft.fft(y1)[: self.freq_hz.size] * dt)
                yf2 = np.abs(np.fft.fft(y2)[: self.freq_hz.size] * dt)
                yfs = np.abs(np.fft.fft(ysum)[: self.freq_hz.size] * dt)
                yfr = np.abs(np.fft.fft(rterm)[: self.freq_hz.size] * dt)
                self.ax.loglog(self.freq_hz[1:], yf1[1:], color="tab:blue", linestyle="--", label="L1 FFT")
                self.ax.loglog(self.freq_hz[1:], yf2[1:], color="tab:orange", linestyle="--", label="L2 FFT")
                self.ax.loglog(self.freq_hz[1:], yfs[1:], color="black", linestyle=":", label="(L1+L2) FFT")
                self.ax.loglog(self.freq_hz[1:], yfr[1:], color="tab:green", linestyle="-.", label="r-term FFT")
            else:
                yf_all = np.fft.fft(y_series) * dt
                yf = np.abs(yf_all[: self.freq_hz.size])
                self.ax.loglog(self.freq_hz[1:], yf[1:], color="tab:orange", linestyle="--", label="FFT(time response)")
            if idx >= 0 and var_name != "L_terms":
                ym = np.abs(self.h_f_matrix[idx])
                self.ax.loglog(self.freq_hz[1:], ym[1:], color="tab:blue", label="Matrix responsivity")
            self.ax.set_xlabel("Frequency [Hz]")
            self.ax.set_ylabel(f"|{var_name}(f)| [{unit_f}]")
            self.ax.set_title(f"Ho Event Response: {var_name}(f)")
            self.ax.legend(loc="best")
        self.ax.grid(True, which="both", alpha=0.25)
        self.canvas.draw_idle()
        err = np.nan
        if self.domain_var.get() == "Frequency" and idx >= 0:
            denom = np.maximum(np.abs(self.h_f_matrix[idx, 1:]), 1.0e-30)
            err = float(np.median(np.abs(np.abs(self.y_fft[idx, 1:]) - np.abs(self.h_f_matrix[idx, 1:])) / denom))
        if np.isfinite(err):
            info = getattr(self, "_time_grid_info", None)
            if isinstance(info, dict):
                clip_txt = " (grid clipped)" if bool(info.get("clipped", False)) else ""
                self.status.set(
                    f"Median |FFT-matrix|/matrix: {err:.3e} | N={int(info.get('n_used', 0))}, "
                    f"tau_min={float(info.get('tau_min_s', np.nan)):.3e}s, tau_max={float(info.get('tau_max_s', np.nan)):.3e}s{clip_txt}"
                )
            else:
                self.status.set(f"Median |FFT-matrix|/matrix: {err:.3e}")
        else:
            self.status.set("")


if __name__ == "__main__":
    main()

