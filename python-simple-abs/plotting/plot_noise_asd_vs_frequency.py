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
            "nep_sufficiency_percent",
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
            "tls_phi_asd_100hz_per_rtHz",
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
        ],
    ),
    (
        "KID2 / Island2",
        [
            "T02_K",
            "heat_capacity2_eV_per_mK",
            "G2_W_per_K",
            "alpha_A2",
            "alpha_phi2",
            "series_L2_H",
            "series_R2_Ohm",
            "feedback_heater_gain_W_per_rad",
        ],
    ),
]
INPUT_KEYS = [k for _, keys in INPUT_SECTIONS for k in keys]
KID2_KEYS = (
    "T02_K",
    "heat_capacity2_eV_per_mK",
    "G2_W_per_K",
    "alpha_A2",
    "alpha_phi2",
    "series_L2_H",
    "series_R2_Ohm",
    "feedback_heater_gain_W_per_rad",
)


def _all_kid2_zero(settings: dict[str, float]) -> bool:
    return all(abs(float(settings.get(k, 0.0))) == 0.0 for k in KID2_KEYS)

LABELS = {
    "T0_K": "T0 [K]",
    "Tb_K": "Tb [K]",
    "pg_drive_dBm": "Drive [dBm]",
    "f0_Hz": "f0 [Hz]",
    "detuning_widths": "Detuning [widths]",
    "nep_sufficiency_percent": "NEP suff [%]",
    "heat_capacity_eV_per_mK": "C [eV/mK]",
    "ho_in_au_atomic_fraction": "Ho/Au frac",
    "tls_phi_asd_100hz_per_rtHz": "TLS ASD @100Hz",
    "tls_beta": "TLS beta",
    "Qi": "Qi",
    "Qc": "Qc",
    "tau_qp_s": "tau_qp [s]",
    "kinetic_inductance_fraction": "kinetic frac",
    "alpha_A": "alpha_A",
    "alpha_phi": "alpha_phi",
    "T02_K": "T02 [K]",
    "heat_capacity2_eV_per_mK": "C2 [eV/mK]",
    "G2_W_per_K": "G2 [W/K]",
    "alpha_A2": "alpha_A2",
    "alpha_phi2": "alpha_phi2",
    "series_L2_H": "L2 [H]",
    "series_R2_Ohm": "R2 [Ohm]",
    "feedback_heater_gain_W_per_rad": "Kfb [W/rad]",
}


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
        self.current, self.last_loaded, self.last_loaded_name = self._load_startup_settings()
        self.undo_stack: list[dict[str, float]] = []

        self.root = tk.Tk()
        self.root.title("Noise ASD / NEP Plotter")
        self.root.geometry("1500x900")

        self.mode_var = tk.StringVar(value="Noise ASD")
        self.readout_var = tk.StringVar(value="Phase")
        kid2_active_default = any(abs(float(self.current.get(k, 0.0))) > 0.0 for k in KID2_KEYS)
        self.kid2_mode_var = tk.StringVar(value="Dual KID" if kid2_active_default else "Single KID")
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.loaded_name_var = tk.StringVar(value="")
        self.entry_vars: dict[str, tk.StringVar] = {}
        self.entry_widgets: dict[str, ttk.Entry] = {}

        self._build_layout()
        self._recompute_and_draw()

    @staticmethod
    def _settings_filetypes() -> list[tuple[str, str]]:
        return [("Pickle files", "*.pkl"), ("All files", "*.*")]

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew")
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

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=1, sticky="ns")

        ttk.Label(controls, text="Noise/NEP Controls", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, textvariable=self.loaded_name_var, foreground="#222").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        mode_frame = ttk.LabelFrame(controls, text="Mode", padding=6)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="Noise ASD", variable=self.mode_var, value="Noise ASD", command=self._on_mode).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(mode_frame, text="NEP", variable=self.mode_var, value="NEP", command=self._on_mode).grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )

        readout_frame = ttk.LabelFrame(controls, text="Readout", padding=6)
        readout_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            readout_frame, text="Phase", variable=self.readout_var, value="Phase", command=self._on_mode
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            readout_frame, text="Amplitude", variable=self.readout_var, value="Amplitude", command=self._on_mode
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        kid_mode_frame = ttk.LabelFrame(controls, text="Configuration", padding=6)
        kid_mode_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            kid_mode_frame, text="Single KID", variable=self.kid2_mode_var, value="Single KID", command=self._on_kid2_mode
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            kid_mode_frame, text="Dual KID", variable=self.kid2_mode_var, value="Dual KID", command=self._on_kid2_mode
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        row = 5
        for section_name, keys in INPUT_SECTIONS:
            section = ttk.LabelFrame(controls, text=section_name, padding=6)
            section.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            row += 1
            for i, key in enumerate(keys):
                ttk.Label(section, text=LABELS[key]).grid(row=i, column=0, sticky="w", padx=(0, 6), pady=2)
                var = tk.StringVar(value=f"{self.current[key]:.8g}")
                ent = ttk.Entry(section, textvariable=var, width=16)
                ent.grid(row=i, column=1, sticky="ew", pady=2)
                ent.bind("<Return>", self._on_field_commit)
                ent.bind("<FocusOut>", self._on_field_commit)
                self.entry_vars[key] = var
                self.entry_widgets[key] = ent

        self._set_kid2_fields_enabled()

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        row += 1

        ttk.Button(button_frame, text="Defaults", command=self._load_defaults).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Undo", command=self._undo).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Save", command=self._save_current).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Load", command=self._load_saved).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Restore", command=self._restore_last_loaded).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(button_frame, text="Match KID2->KID1", command=self._match_kid2_to_kid1).grid(
            row=1, column=2, padx=2, pady=2, sticky="ew"
        )

        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        ttk.Label(controls, textvariable=self.status_var, wraplength=300, foreground="#444").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        row += 1

        ttk.Label(
            controls,
            textvariable=self.summary_var,
            justify="right",
            anchor="e",
            wraplength=320,
            foreground="#222",
        ).grid(row=row, column=0, columnspan=2, sticky="e", pady=(14, 0))

    def _load_saved_or_defaults(self) -> dict[str, float]:
        if SETTINGS_FILE.exists():
            try:
                with SETTINGS_FILE.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    merged = dict(self.defaults)
                    for k in INPUT_KEYS:
                        if k in loaded:
                            merged[k] = float(loaded[k])
                    return merged
            except Exception:
                pass
        return dict(self.defaults)

    def _load_settings_file(self, path: Path) -> dict[str, float] | None:
        try:
            with path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                return None
            vals = dict(self.defaults)
            for k in INPUT_KEYS:
                if k in loaded:
                    vals[k] = float(loaded[k])
            return vals
        except Exception:
            return None

    def _load_startup_settings(self) -> tuple[dict[str, float], dict[str, float], str | None]:
        if STARTUP_STATE_FILE.exists():
            try:
                with STARTUP_STATE_FILE.open("rb") as f:
                    state = pickle.load(f)
                if isinstance(state, dict):
                    source = state.get("source_path")
                    if isinstance(source, str) and source:
                        source_path = Path(source)
                        vals = self._load_settings_file(source_path)
                        if vals is not None:
                            return vals, dict(vals), source_path.name
            except Exception:
                pass

        vals = self._load_saved_or_defaults()
        name = SETTINGS_FILE.name if SETTINGS_FILE.exists() else None
        return vals, dict(vals), name

    def _persist_startup_state(self, source_path: Path) -> None:
        try:
            with STARTUP_STATE_FILE.open("wb") as f:
                pickle.dump({"source_path": str(source_path)}, f)
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
        heater2_pW = 1.0e12 * float(s.heater2_dc_power_W)
        self.summary_var.set(
            "Event dT (island): "
            f"{delta_t_mk:.3g} mK\n"
            "Average event rate: "
            f"{rate_hz:.3g} Hz\n"
            "Pileup probability: "
            f"{pileup_pct:.3g}%\n"
            "Pulse shortening factor: "
            f"{shorten:.3g}\n"
            "KID2 heater power: "
            f"{heater2_pW:.3g} pW"
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
        if self.kid2_mode_var.get() == "Single KID":
            for k in KID2_KEYS:
                kwargs[k] = 0.0
        return Sensor(Version1SensorInputs(**kwargs))

    def _set_kid2_fields_enabled(self) -> None:
        dual = self.kid2_mode_var.get() == "Dual KID"
        state = "normal" if dual else "disabled"
        for k in KID2_KEYS:
            w = self.entry_widgets.get(k)
            if w is not None:
                w.configure(state=state)

    def _populate_kid2_from_kid1(self, g1: float, l1: float, r1: float) -> None:
        """Initialize KID2 parameters to a nonzero mirror of KID1-side values."""
        if not _all_kid2_zero(self.current):
            return
        base = dict(self.current)
        base["T02_K"] = float(base.get("T0_K", 0.04))
        base["heat_capacity2_eV_per_mK"] = float(base.get("heat_capacity_eV_per_mK", 150.0))
        base["G2_W_per_K"] = float(g1)
        base["alpha_A2"] = float(base.get("alpha_A", 0.1))
        base["alpha_phi2"] = float(base.get("alpha_phi", 140.0))
        base["series_L2_H"] = float(l1)
        base["series_R2_Ohm"] = float(r1)
        base["feedback_heater_gain_W_per_rad"] = 0.0
        self.current = base
        self._write_fields(self.current)

    def _on_kid2_mode(self) -> None:
        if self.kid2_mode_var.get() == "Dual KID":
            try:
                # Sync from latest user-edited values before mirroring.
                self.current = self._read_fields()
            except Exception:
                pass
            # Build a temporary sensor so derived mirror quantities are available.
            g1 = 0.0
            l1 = 0.0
            r1 = 0.0
            try:
                s = self._build_sensor(self.current)
                g1 = float(s.G_W_per_K)
                l1 = float(s.L_total_H)
                r1 = float(s.R1_series_Ohm)
            except Exception:
                pass
            self._populate_kid2_from_kid1(g1=g1, l1=l1, r1=r1)
        self._set_kid2_fields_enabled()
        self._recompute_and_draw()

    def _match_kid2_to_kid1(self) -> None:
        try:
            self.current = self._read_fields()
            s = self._build_sensor(self.current)
            self.current["T02_K"] = float(self.current.get("T0_K", 0.04))
            self.current["heat_capacity2_eV_per_mK"] = float(self.current.get("heat_capacity_eV_per_mK", 150.0))
            self.current["G2_W_per_K"] = float(s.G_W_per_K)
            self.current["alpha_A2"] = float(self.current.get("alpha_A", 0.1))
            self.current["alpha_phi2"] = float(self.current.get("alpha_phi", 140.0))
            self.current["series_L2_H"] = float(s.L_total_H)
            self.current["series_R2_Ohm"] = float(s.R1_series_Ohm)
            self.current["feedback_heater_gain_W_per_rad"] = 0.0
            self.kid2_mode_var.set("Dual KID")
            self._write_fields(self.current)
            self._set_kid2_fields_enabled()
            self._recompute_and_draw()
            self._set_status("KID2 parameters matched to KID1")
        except Exception as exc:
            self._set_status(f"Match KID2->KID1 failed: {exc}")

    def _compute_data(self, s: Sensor) -> dict[str, object]:
        eigs = np.array(s.mt_eigenvalues, dtype=complex)
        max_mode_rate_per_s = float(np.max(np.abs(eigs)))
        f_min_hz = 0.1
        f_max_hz = max(f_min_hz * 10.0, 10.0 * max_mode_rate_per_s / (2.0 * pi))
        freqs_hz = np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), 1000)

        asd_phase_johnson = np.zeros_like(freqs_hz)
        asd_phase_johnson_2 = np.zeros_like(freqs_hz)
        asd_phase_phonon = np.zeros_like(freqs_hz)
        asd_phase_phonon_2 = np.zeros_like(freqs_hz)
        asd_phase_tls = np.zeros_like(freqs_hz)
        asd_phase_electronic = np.zeros_like(freqs_hz)
        asd_phase_electronic_2 = np.zeros_like(freqs_hz)
        asd_amp_johnson = np.zeros_like(freqs_hz)
        asd_amp_johnson_2 = np.zeros_like(freqs_hz)
        asd_amp_phonon = np.zeros_like(freqs_hz)
        asd_amp_phonon_2 = np.zeros_like(freqs_hz)
        asd_amp_tls = np.zeros_like(freqs_hz)
        asd_amp_electronic = np.zeros_like(freqs_hz)
        asd_amp_electronic_2 = np.zeros_like(freqs_hz)
        phase_resp = np.zeros_like(freqs_hz)
        amp_resp = np.zeros_like(freqs_hz)

        for i, f_hz in enumerate(freqs_hz):
            y_j_a1 = s._propagate_noise_vector(s.n_johnson_A_1(), f_hz)
            y_j_phi1 = s._propagate_noise_vector(s.n_johnson_phi_1(), f_hz)
            y_j_a_2 = s._propagate_noise_vector(s.n_johnson_A_2(), f_hz)
            y_j_phi_2 = s._propagate_noise_vector(s.n_johnson_phi_2(), f_hz)
            y_ph1 = s._propagate_noise_vector(s.n_phonon_1(), f_hz)
            y_ph_2 = s._propagate_noise_vector(s.n_phonon_2(), f_hz)
            y_tls = s._propagate_noise_vector(s.n_tls_phi_at_hz(float(f_hz)), f_hz)
            y_e_a1 = s._propagate_noise_vector(s.n_electronic_A_1(), f_hz)
            y_e_phi1 = s._propagate_noise_vector(s.n_electronic_phi_1(), f_hz)
            y_e_a_2 = s._propagate_noise_vector(s.n_electronic_A_2(), f_hz)
            y_e_phi_2 = s._propagate_noise_vector(s.n_electronic_phi_2(), f_hz)

            asd_phase_johnson[i] = np.sqrt(abs(y_j_a1[1]) ** 2 + abs(y_j_phi1[1]) ** 2 + abs(y_j_a_2[1]) ** 2 + abs(y_j_phi_2[1]) ** 2)
            asd_phase_johnson_2[i] = np.sqrt(abs(y_j_a_2[1]) ** 2 + abs(y_j_phi_2[1]) ** 2)
            asd_phase_phonon[i] = np.sqrt(abs(y_ph1[1]) ** 2 + abs(y_ph_2[1]) ** 2)
            asd_phase_phonon_2[i] = abs(y_ph_2[1])
            asd_phase_tls[i] = abs(y_tls[1])
            asd_phase_electronic[i] = np.sqrt(abs(y_e_a1[1]) ** 2 + abs(y_e_phi1[1]) ** 2 + abs(y_e_a_2[1]) ** 2 + abs(y_e_phi_2[1]) ** 2)
            asd_phase_electronic_2[i] = np.sqrt(abs(y_e_a_2[1]) ** 2 + abs(y_e_phi_2[1]) ** 2)
            asd_amp_johnson[i] = np.sqrt(abs(y_j_a1[0]) ** 2 + abs(y_j_phi1[0]) ** 2 + abs(y_j_a_2[0]) ** 2 + abs(y_j_phi_2[0]) ** 2)
            asd_amp_johnson_2[i] = np.sqrt(abs(y_j_a_2[0]) ** 2 + abs(y_j_phi_2[0]) ** 2)
            asd_amp_phonon[i] = np.sqrt(abs(y_ph1[0]) ** 2 + abs(y_ph_2[0]) ** 2)
            asd_amp_phonon_2[i] = abs(y_ph_2[0])
            asd_amp_tls[i] = abs(y_tls[0])
            asd_amp_electronic[i] = np.sqrt(abs(y_e_a1[0]) ** 2 + abs(y_e_phi1[0]) ** 2 + abs(y_e_a_2[0]) ** 2 + abs(y_e_phi_2[0]) ** 2)
            asd_amp_electronic_2[i] = np.sqrt(abs(y_e_a_2[0]) ** 2 + abs(y_e_phi_2[0]) ** 2)
            phase_resp[i] = s.phase_responsivity_mag_rad_per_W_at_hz(float(f_hz))
            y_unit_power = np.linalg.solve(
                s.m_matrix_array(float(f_hz)),
                np.array((0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j), dtype=complex),
            )
            amp_resp[i] = abs(y_unit_power[0])

        asd_phase_total = np.sqrt(asd_phase_johnson**2 + asd_phase_phonon**2 + asd_phase_tls**2 + asd_phase_electronic**2)
        asd_amp_total = np.sqrt(asd_amp_johnson**2 + asd_amp_phonon**2 + asd_amp_tls**2 + asd_amp_electronic**2)
        asd_tls_direct = s.tls_phi_asd_100hz_per_rtHz * ((freqs_hz / 100.0) ** (-s.tls_beta / 2.0))
        asd_johnson_simple = abs(s.f0_Hz * s.dphi_df_detuning_per_hz) * np.sqrt(s.sf_over_f0sq_johnson_simple)
        nep_phase_johnson = np.where(phase_resp > 0.0, asd_phase_johnson / phase_resp, np.nan)
        nep_phase_johnson_2 = np.where(phase_resp > 0.0, asd_phase_johnson_2 / phase_resp, np.nan)
        nep_phase_phonon = np.where(phase_resp > 0.0, asd_phase_phonon / phase_resp, np.nan)
        nep_phase_phonon_2 = np.where(phase_resp > 0.0, asd_phase_phonon_2 / phase_resp, np.nan)
        nep_phase_tls = np.where(phase_resp > 0.0, asd_phase_tls / phase_resp, np.nan)
        nep_phase_electronic = np.where(phase_resp > 0.0, asd_phase_electronic / phase_resp, np.nan)
        nep_phase_electronic_2 = np.where(phase_resp > 0.0, asd_phase_electronic_2 / phase_resp, np.nan)
        nep_phase_total = np.where(phase_resp > 0.0, asd_phase_total / phase_resp, np.nan)
        nep_amp_johnson = np.where(amp_resp > 0.0, asd_amp_johnson / amp_resp, np.nan)
        nep_amp_johnson_2 = np.where(amp_resp > 0.0, asd_amp_johnson_2 / amp_resp, np.nan)
        nep_amp_phonon = np.where(amp_resp > 0.0, asd_amp_phonon / amp_resp, np.nan)
        nep_amp_phonon_2 = np.where(amp_resp > 0.0, asd_amp_phonon_2 / amp_resp, np.nan)
        nep_amp_tls = np.where(amp_resp > 0.0, asd_amp_tls / amp_resp, np.nan)
        nep_amp_electronic = np.where(amp_resp > 0.0, asd_amp_electronic / amp_resp, np.nan)
        nep_amp_electronic_2 = np.where(amp_resp > 0.0, asd_amp_electronic_2 / amp_resp, np.nan)
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
            [asd_phase_johnson, asd_phase_phonon, asd_phase_tls, asd_phase_electronic, asd_phase_total, asd_tls_direct, np.array([asd_johnson_simple])]
        )
        asd_amp_ylim = _positive_limits([asd_amp_johnson, asd_amp_phonon, asd_amp_tls, asd_amp_electronic, asd_amp_total])
        nep_phase_ylim = _positive_limits([nep_phase_johnson, nep_phase_phonon, nep_phase_tls, nep_phase_electronic, nep_phase_total])
        nep_amp_ylim = _positive_limits([nep_amp_johnson, nep_amp_phonon, nep_amp_tls, nep_amp_electronic, nep_amp_total])

        return {
            "sensor": s,
            "freqs": freqs_hz,
            "asd_phase": (asd_phase_johnson, asd_phase_phonon, asd_phase_tls, asd_phase_electronic, asd_phase_total),
            "asd_phase_2": (asd_phase_johnson_2, asd_phase_phonon_2, asd_phase_electronic_2),
            "asd_amp": (asd_amp_johnson, asd_amp_phonon, asd_amp_tls, asd_amp_electronic, asd_amp_total),
            "asd_amp_2": (asd_amp_johnson_2, asd_amp_phonon_2, asd_amp_electronic_2),
            "asd_tls_direct": asd_tls_direct,
            "nep_phase": (nep_phase_johnson, nep_phase_phonon, nep_phase_tls, nep_phase_electronic, nep_phase_total),
            "nep_phase_2": (nep_phase_johnson_2, nep_phase_phonon_2, nep_phase_electronic_2),
            "nep_amp": (nep_amp_johnson, nep_amp_phonon, nep_amp_tls, nep_amp_electronic, nep_amp_total),
            "nep_amp_2": (nep_amp_johnson_2, nep_amp_phonon_2, nep_amp_electronic_2),
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
        asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total = d["asd_phase"] if is_phase else d["asd_amp"]
        asd_johnson_2, asd_phonon_2, asd_electronic_2 = d["asd_phase_2"] if is_phase else d["asd_amp_2"]
        nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total = d["nep_phase"] if is_phase else d["nep_amp"]
        nep_johnson_2, nep_phonon_2, nep_electronic_2 = d["nep_phase_2"] if is_phase else d["nep_amp_2"]

        self.ax.clear()
        mode = self.mode_var.get()
        dual = self.kid2_mode_var.get() == "Dual KID"
        if mode == "NEP":
            ysets = (nep_johnson, nep_phonon, nep_tls, nep_electronic, nep_total)
            branch1 = (
                np.sqrt(np.maximum(nep_johnson**2 - nep_johnson_2**2, 0.0)),
                np.sqrt(np.maximum(nep_phonon**2 - nep_phonon_2**2, 0.0)),
                nep_tls,
                np.sqrt(np.maximum(nep_electronic**2 - nep_electronic_2**2, 0.0)),
            )
            branch2 = (nep_johnson_2, nep_phonon_2, nep_electronic_2)
            ylab = "NEP [W/rtHz]"
            title = f"Noise-Equivalent Power vs Frequency ({readout} readout)"
            self.ax.set_ylim(*(d["nep_phase_ylim"] if is_phase else d["nep_amp_ylim"]))
        else:
            ysets = (asd_johnson, asd_phonon, asd_tls, asd_electronic, asd_total)
            branch1 = (
                np.sqrt(np.maximum(asd_johnson**2 - asd_johnson_2**2, 0.0)),
                np.sqrt(np.maximum(asd_phonon**2 - asd_phonon_2**2, 0.0)),
                asd_tls,
                np.sqrt(np.maximum(asd_electronic**2 - asd_electronic_2**2, 0.0)),
            )
            branch2 = (asd_johnson_2, asd_phonon_2, asd_electronic_2)
            ylab = "Phase ASD [rad/rtHz]" if is_phase else "Amplitude ASD [1/rtHz]"
            title = f"Noise ASD vs Frequency ({readout} readout)"
            self.ax.set_ylim(*(d["asd_phase_ylim"] if is_phase else d["asd_amp_ylim"]))

        if dual:
            self.ax.loglog(freqs, branch1[0], label="Johnson (KID1)", color="tab:blue")
            self.ax.loglog(freqs, branch1[1], label="Phonon (KID1)", color="tab:orange")
            self.ax.loglog(freqs, branch1[2], label="TLS", color="tab:green")
            self.ax.loglog(freqs, branch1[3], label="Electronic (KID1)", color="tab:red")
            self.ax.loglog(freqs, branch2[0], linestyle="--", linewidth=2.0, color="#8fb7ff", alpha=0.98, label="Johnson (KID2)")
            self.ax.loglog(freqs, branch2[1], linestyle="--", linewidth=2.0, color="#ffd59a", alpha=0.98, label="Phonon (KID2)")
            self.ax.loglog(freqs, branch2[2], linestyle="--", linewidth=2.0, color="#ffb3b3", alpha=0.98, label="Electronic (KID2)")
        else:
            self.ax.loglog(freqs, ysets[0], label="Johnson", color="tab:blue")
            self.ax.loglog(freqs, ysets[1], label="Phonon", color="tab:orange")
            self.ax.loglog(freqs, ysets[2], label="TLS", color="tab:green")
            self.ax.loglog(freqs, ysets[3], label="Electronic", color="tab:red")
        self.ax.loglog(freqs, ysets[4], color="k", linestyle=":", linewidth=2.2, label="Total (quadrature)")

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
            y_thr = float(ysets[4][idx])
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
        self._draw()
        if self.kid2_mode_var.get() == "Dual KID":
            no_kid2_noise = (
                abs(sensor.nj2_scale) == 0.0
                and abs(sensor.nj2_thermal_scale) == 0.0
                and abs(sensor.G2_W_per_K) == 0.0
            )
            if no_kid2_noise:
                self._set_status(
                    "Dual KID selected, but KID2 noise is zero. Set series_R2_Ohm (>0) for Johnson/electronic and G2_W_per_K (>0) for phonon."
                )

    def _on_mode(self) -> None:
        self._draw()

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
            with save_path.open("wb") as f:
                pickle.dump({k: self.current[k] for k in INPUT_KEYS}, f)
            self.last_loaded = dict(self.current)
            self.last_loaded_name = save_path.name
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
            vals = self._load_settings_file(load_path)
            if vals is None:
                raise ValueError("invalid settings file")
            prev = dict(self.current)
            self.current = vals
            self.last_loaded = dict(vals)
            self.last_loaded_name = load_path.name
            self._persist_startup_state(load_path)
            self._write_fields(self.current)
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status(f"Loaded settings from {load_path.name}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")

    def _restore_last_loaded(self) -> None:
        prev = dict(self.current)
        self.current = dict(self.last_loaded)
        self._write_fields(self.current)
        try:
            self._recompute_and_draw()
            self._push_undo(prev)
            self._update_loaded_name()
            self._set_status("Restored last loaded settings")
        except Exception as exc:
            self.current = prev
            self._write_fields(self.current)
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

    def run(self) -> None:
        startup = self.last_loaded_name if self.last_loaded_name else "defaults"
        self._update_loaded_name()
        self._set_status(f"Loaded startup settings ({startup})")
        self.root.mainloop()


def main() -> None:
    app = NoiseGui()
    app.run()


if __name__ == "__main__":
    main()
