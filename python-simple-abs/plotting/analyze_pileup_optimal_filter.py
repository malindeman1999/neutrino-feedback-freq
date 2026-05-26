"""Analyze saved phi pulse sweeps with a single-pulse optimal filter."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pileup_sweep import analyze_pulse_sweep_dataset


PLOT_DIR = Path(__file__).resolve().parent
SAVES_DIR = PLOT_DIR / "saves"
ANALYZER_STATE_FILE = SAVES_DIR / "analyze_pileup_gui_state.pkl"


def _scatter_by_class(
    ax: plt.Axes,
    results: dict[str, dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    norm: Normalize,
) -> None:
    single = results["single"]
    double = results["double"]
    ax.scatter(
        single[x_key],
        single[y_key],
        c=single["lag_s"] * 1.0e6,
        norm=norm,
        cmap="viridis",
        marker="o",
        linewidths=0.8,
        label="Single pulse",
    )
    pts = ax.scatter(
        double[x_key],
        double[y_key],
        c=double["lag_s"] * 1.0e6,
        norm=norm,
        cmap="viridis",
        marker="x",
        linewidths=1.0,
        label="Double pulse",
    )
    return pts


def _plot_inputs(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    x_single = np.asarray(results["single"]["true_total_energy_eV"], dtype=float)
    x_double = np.min(np.asarray(dataset["double"]["energies_eV"], dtype=float), axis=1)
    x_map = {"single": x_single, "double": x_double}
    results_plot = {
        "single": dict(results["single"]),
        "double": dict(results["double"]),
    }
    results_plot["single"]["min_pulse_energy_eV"] = x_map["single"]
    results_plot["double"]["min_pulse_energy_eV"] = x_map["double"]
    return results_plot, x_single, x_double


def make_energy_closure_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    results_plot, x_single, x_double = _plot_inputs(dataset, results)
    lag_max_us = max(float(np.max(results["double"]["lag_s"])) * 1.0e6, 1.0e-12)
    norm = Normalize(vmin=0.0, vmax=lag_max_us)
    fig, ax_energy = plt.subplots(1, 1, figsize=(6.2, 5.0), constrained_layout=True)

    points = _scatter_by_class(ax_energy, results_plot, "min_pulse_energy_eV", "fit_energy_eV", norm)
    all_x = np.concatenate((x_single, x_double))
    x_min = float(np.min(all_x))
    x_max = float(np.max(all_x))
    y_all = np.concatenate((results["single"]["fit_energy_eV"], results["double"]["fit_energy_eV"]))
    y_pos = y_all[y_all > 0.0]
    if y_pos.size:
        ax_energy.set_ylim(float(np.min(y_pos)) * 0.8, float(np.max(y_pos)) * 1.25)
    ax_energy.set_xlim(max(0.0, x_min * 0.8), x_max * 1.25)
    ax_energy.plot([max(0.0, x_min), x_max], [max(0.0, x_min), x_max], color="black", linestyle="--", linewidth=1.0, label="Expected")
    ax_energy.set_xlabel("Minimum pulse energy [eV]")
    ax_energy.set_ylabel("Single-pulse filter energy [eV]")
    ax_energy.set_title("Optimal-Filter Energy Closure")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(loc="best")
    cbar = fig.colorbar(points, ax=ax_energy, label="Pulse separation [us]")
    cbar.ax.set_ylabel("Pulse separation [us]")
    tau_s = float(dataset["generation"]["tau_decay_s"])
    fig.suptitle(f"Phi optimal-filter study; pulse decay tau = {tau_s * 1.0e3:.3g} ms")
    return fig


def make_energy_vs_chi2_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    results_plot, _, _ = _plot_inputs(dataset, results)
    lag_max_us = max(float(np.max(results["double"]["lag_s"])) * 1.0e6, 1.0e-12)
    norm = Normalize(vmin=0.0, vmax=lag_max_us)
    fig, ax_chi2 = plt.subplots(1, 1, figsize=(6.2, 5.0), constrained_layout=True)
    points = _scatter_by_class(ax_chi2, results_plot, "min_pulse_energy_eV", "reduced_chi2", norm)
    ax_chi2.set_xlabel("Minimum pulse energy [eV]")
    ax_chi2.set_ylabel("Single-pulse fit reduced chi2")
    ax_chi2.set_title("Pile-Up Discrimination Statistic")
    ax_chi2.set_xscale("log")
    ax_chi2.set_yscale("log")
    ax_chi2.grid(True, which="both", alpha=0.25)
    ax_chi2.legend(loc="best")
    cbar = fig.colorbar(points, ax=ax_chi2, label="Pulse separation [us]")
    cbar.ax.set_ylabel("Pulse separation [us]")
    tau_s = float(dataset["generation"]["tau_decay_s"])
    fig.suptitle(f"Phi optimal-filter study; pulse decay tau = {tau_s * 1.0e3:.3g} ms")
    return fig


def make_lag_vs_chi2_figure(dataset: dict, results: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    results_plot, _, _ = _plot_inputs(dataset, results)
    lag_us_double = np.asarray(results["double"]["lag_s"], dtype=float) * 1.0e6
    energies_double = np.asarray(dataset["double"]["energies_eV"], dtype=float)
    min_energy_double = np.min(energies_double, axis=1)
    max_energy_double = np.max(energies_double, axis=1)
    min_energy_single = np.asarray(results_plot["single"]["min_pulse_energy_eV"], dtype=float)
    unique_min = np.sort(np.unique(min_energy_double))[::-1]
    nrows = int(unique_min.size)
    fig, axes = plt.subplots(1 if nrows == 1 else nrows, 1, figsize=(7.2, max(3.2, 2.45 * nrows)), constrained_layout=True)
    axes_arr = [axes] if nrows == 1 else list(axes)
    big_norm = Normalize(vmin=float(np.min(max_energy_double)), vmax=float(np.max(max_energy_double)))
    sc_for_cbar = None
    for i, e_min in enumerate(unique_min):
        ax = axes_arr[i]
        mask = min_energy_double == e_min
        sc_for_cbar = ax.scatter(
            lag_us_double[mask],
            np.asarray(results["double"]["reduced_chi2"], dtype=float)[mask],
            c=max_energy_double[mask],
            cmap="plasma",
            norm=big_norm,
            marker="x",
            linewidths=1.0,
            label="Double pulse",
        )
        single_mask = np.isclose(min_energy_single, e_min)
        if np.any(single_mask):
            ax.scatter(
                np.zeros(np.count_nonzero(single_mask), dtype=float),
                np.asarray(results["single"]["reduced_chi2"], dtype=float)[single_mask],
                c=min_energy_single[single_mask],
                cmap="plasma",
                norm=big_norm,
                marker="o",
                linewidths=0.8,
                label="Single pulse",
            )
        ax.set_ylabel("Reduced chi2")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        ax.set_title(f"Lag vs Pile-Up Metric (min pulse energy = {e_min:.3g} eV)")
    axes_arr[-1].set_xlabel("Pulse separation [us]")
    if sc_for_cbar is not None:
        cbar_lag = fig.colorbar(sc_for_cbar, ax=axes_arr, label="Larger pulse energy [eV]")
        cbar_lag.ax.set_ylabel("Larger pulse energy [eV]")
    tau_s = float(dataset["generation"]["tau_decay_s"])
    fig.suptitle(f"Phi optimal-filter study; pulse decay tau = {tau_s * 1.0e3:.3g} ms")
    return fig


class AnalyzerGui:
    def __init__(self) -> None:
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        self.root = tk.Tk()
        self.root.title("Pile-Up Optimal Filter Analyzer")
        self.root.geometry("760x260")
        self.dataset_var = tk.StringVar(value=state.get("last_dataset_path", ""))
        self.results_var = tk.StringVar(value="")
        self.save_results_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select a pulse sweep dataset and run analysis.")
        self.results_entry: ttk.Entry | None = None
        self.results_button: ttk.Button | None = None
        self._build()

    @staticmethod
    def _load_state() -> dict[str, str]:
        if not ANALYZER_STATE_FILE.exists():
            return {}
        try:
            with ANALYZER_STATE_FILE.open("rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                return {}
            out: dict[str, str] = {}
            if isinstance(raw.get("last_dataset_path"), str):
                out["last_dataset_path"] = raw["last_dataset_path"]
            return out
        except Exception:
            return {}

    def _save_state(self) -> None:
        state = {"last_dataset_path": self.dataset_var.get().strip()}
        try:
            with ANALYZER_STATE_FILE.open("wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    @staticmethod
    def _filetypes_pickle() -> list[tuple[str, str]]:
        return [("Pickle files", "*.pkl"), ("All files", "*.*")]

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input dataset (.pkl):").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=self.dataset_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse", command=self._pick_dataset).grid(row=0, column=2, sticky="ew", pady=4)

        ttk.Checkbutton(
            frame, text="Save fit results", variable=self.save_results_var, command=self._on_toggle_save
        ).grid(row=1, column=0, sticky="w", pady=4)
        self.results_entry = ttk.Entry(frame, textvariable=self.results_var)
        self.results_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.results_button = ttk.Button(frame, text="Browse", command=self._pick_results)
        self.results_button.grid(row=1, column=2, sticky="ew", pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        buttons.columnconfigure(3, weight=1)
        ttk.Button(buttons, text="Energy Closure", command=self._run_energy_closure).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Energy vs Chi2", command=self._run_energy_vs_chi2).grid(
            row=0, column=1, sticky="ew", padx=(4, 4)
        )
        ttk.Button(buttons, text="Lag vs Chi2", command=self._run_lag_vs_chi2).grid(
            row=0, column=2, sticky="ew", padx=(4, 4)
        )
        ttk.Button(buttons, text="Close", command=self.root.destroy).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        ttk.Label(frame, textvariable=self.status_var, wraplength=730, foreground="#333").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        self._on_toggle_save()

    def _pick_dataset(self) -> None:
        current = self.dataset_var.get().strip()
        initialdir = str(Path(current).parent) if current else str(SAVES_DIR)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select pulse sweep dataset",
            initialdir=initialdir,
            initialfile=(Path(current).name if current else ""),
            filetypes=self._filetypes_pickle(),
        )
        if path:
            self.dataset_var.set(path)
            self._save_state()

    def _pick_results(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save analysis results",
            initialdir=SAVES_DIR,
            initialfile="pileup_optimal_filter_results.pkl",
            defaultextension=".pkl",
            filetypes=self._filetypes_pickle(),
        )
        if path:
            self.results_var.set(path)

    def _on_toggle_save(self) -> None:
        results_state = tk.NORMAL if self.save_results_var.get() else tk.DISABLED
        if self.results_entry is not None:
            self.results_entry.configure(state=results_state)
        if self.results_button is not None:
            self.results_button.configure(state=results_state)

    def _resolve_paths(self) -> tuple[Path | None, Path | None]:
        dataset_path = Path(self.dataset_var.get().strip()) if self.dataset_var.get().strip() else None
        if dataset_path is None or not dataset_path.exists():
            self.status_var.set("Choose a valid input dataset file.")
            return None, None
        results_path = None
        if self.save_results_var.get():
            res_text = self.results_var.get().strip()
            if not res_text:
                self.status_var.set("Choose a results output path or uncheck Save fit results.")
                return None, None
            results_path = Path(res_text)
        return dataset_path, results_path

    def _run_plot(self, plot_kind: str) -> None:
        resolved = self._resolve_paths()
        if resolved[0] is None:
            return
        dataset_path, results_path = resolved
        try:
            self.status_var.set("Running analysis...")
            self.root.update_idletasks()
            with dataset_path.open("rb") as f:
                dataset = pickle.load(f)
            self._save_state()
            results = analyze_pulse_sweep_dataset(dataset)
            if plot_kind == "energy_closure":
                fig = make_energy_closure_figure(dataset, results)
            elif plot_kind == "energy_vs_chi2":
                fig = make_energy_vs_chi2_figure(dataset, results)
            else:
                fig = make_lag_vs_chi2_figure(dataset, results)
            if results_path:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with results_path.open("wb") as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.status_var.set("Analysis complete. Plot window opened.")
            plt.show()
        except Exception as exc:
            self.status_var.set(f"Analysis failed: {exc}")

    def _run_energy_closure(self) -> None:
        self._run_plot("energy_closure")

    def _run_energy_vs_chi2(self) -> None:
        self._run_plot("energy_vs_chi2")

    def _run_lag_vs_chi2(self) -> None:
        self._run_plot("lag_vs_chi2")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = AnalyzerGui()
    app.run()


if __name__ == "__main__":
    main()
