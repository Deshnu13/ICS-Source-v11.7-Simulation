from dataclasses import dataclass
from typing import Tuple, Dict, Any
import os
import sys
import platform
import subprocess
import numpy as np
import scipy.constants as const
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba

# Optuna auto-installation check for multi-objective NSGA-II optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[INFO] Optuna හමු නොවීය. ස්වයංක්‍රීයව ස්ථාපනය කරමින් පවතී...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    OPTUNA_AVAILABLE = True


def open_image_file(filename: str):
    """Directly opens the saved image file supporting WSL, Windows, Linux, and macOS."""
    try:
        abs_path = os.path.abspath(filename)
        sys_name = platform.system()
        release = platform.uname().release.lower()
        
        if "microsoft" in release or "wsl" in release:
            win_path_res = subprocess.run(["wslpath", "-w", abs_path], capture_output=True, text=True)
            win_path = win_path_res.stdout.strip()
            if win_path:
                subprocess.run(["explorer.exe", win_path])
            else:
                subprocess.run(["cmd.exe", "/c", "start", abs_path])
        elif sys_name == "Windows":
            subprocess.run(['start', '', abs_path], shell=True)
        elif sys_name == "Darwin":
            subprocess.run(["open", abs_path])
        else:
            try:
                subprocess.run(["xdg-open", abs_path])
            except FileNotFoundError:
                print("[INFO] xdg-open හමු නොවීය. ෆයිල් එක සාර්ථකව සේව් වී ඇත.")
                return
            
        print(f"[SUCCESS] ප්‍රස්තාරය විවෘත කිරීමට විධානය යවන ලදී: {filename}")
    except Exception as e:
        print(f"[ERROR] ප්‍රස්තාරය විවෘත කිරීමේදී දෝෂයක් මතු විය: {e}")


@numba.njit(fastmath=True)
def klein_nishina_ratio(X: float) -> float:
    """Calculates quantum recoil factor via relativistic Klein-Nishina cross-section ratio."""
    if X < 1e-2:
        return 1.0 - X + 1.3 * (X ** 2) - (133.0 / 105.0) * (X ** 3)
    
    one_p_x = 1.0 + X
    one_p_2x = 1.0 + 2.0 * X
    log_1p2x = np.log(one_p_2x)
    
    term1 = (one_p_x / (X ** 3)) * ((2.0 * X * one_p_x) / one_p_2x - log_1p2x)
    term2 = log_1p2x / (2.0 * X)
    term3 = (1.0 + 3.0 * X) / (one_p_2x ** 2)
    
    return max(1e-6, 0.75 * (term1 + term2 - term3))


@numba.njit(fastmath=True)
def compute_masterpiece_spectrum_numba(
    energy_grid: np.ndarray,
    peak_energy: float,
    sigma_energy: float,
    total_yield: float
) -> np.ndarray:
    """Numba-accelerated high-yield self-consistent spectrum kernel."""
    num_e = energy_grid.shape[0]
    spectrum = np.zeros(num_e, dtype=np.float64)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_energy)
    
    for i in range(num_e):
        e_val = energy_grid[i]
        diff = e_val - peak_energy
        val = total_yield * norm * np.exp(-0.5 * ((diff / sigma_energy) ** 2))
        spectrum[i] = max(0.0, val)
        
    return spectrum


@dataclass(frozen=True)
class MasterpieceElectronBeam:
    energy_mev: float = 35.0
    charge_pc: float = 500.0         
    norm_emittance_um: float = 0.005  
    energy_spread_pct: float = 0.000001  
    bunch_length_fs: float = 10.0
    spot_size_um: float = 3.0           
    enable_beta_matching: bool = True     
    enable_space_charge: bool = True      

    @property
    def gamma(self) -> float:
        m_e_ev = (const.m_e * const.c**2) / const.e
        return (self.energy_mev * 1e6) / m_e_ev

    @property
    def num_electrons(self) -> float:
        return (self.charge_pc * 1e-12) / const.e

    @property
    def bunch_length_m(self) -> float:
        return self.bunch_length_fs * 1e-15 * const.c

    @property
    def peak_current_A(self) -> float:
        tau_s = self.bunch_length_fs * 1e-15
        return (self.charge_pc * 1e-12) / tau_s

    @property
    def effective_norm_emittance_um(self) -> float:
        if not self.enable_space_charge:
            return self.norm_emittance_um
        
        alfven_current = 17045.0  
        i_peak = self.peak_current_A
        sigma_z_um = self.bunch_length_m * 1e6
        c_sc = 0.35
        delta_emit_sc = c_sc * (i_peak / (alfven_current * (self.gamma ** 2))) * sigma_z_um
        return np.sqrt(self.norm_emittance_um**2 + delta_emit_sc**2)

    @property
    def divergence_rad(self) -> float:
        spot_m = self.spot_size_um * 1e-6
        eff_emit_m = self.effective_norm_emittance_um * 1e-6
        return eff_emit_m / (self.gamma * spot_m)


@dataclass(frozen=True)
class MasterpieceLaser:
    wavelength_nm: float = 800.0
    pulse_energy_j: float = 0.05         
    pulse_duration_fs: float = 500000.0  
    waist_um: float = 140.0              
    collision_angle_deg: float = 180.0
    pinhole_radius_mm: float = 0.035     
    pinhole_distance_m: float = 2.5    
    
    polarization: str = "circular"             
    pulse_profile: str = "flat_top"            
    enable_slm_correction: bool = True         
    enable_optical_cavity: bool = True         
    cavity_finesse_gain: float = 3000.0  
    
    enable_channel_cut_array: bool = True
    channel_cut_transmission: float = 0.92

    @property
    def m2_factor(self) -> float:
        return 1.00 if self.enable_slm_correction else 1.25

    @property
    def photon_energy_ev(self) -> float:
        wavelength_m = self.wavelength_nm * 1e-9
        return (const.h * const.c) / (wavelength_m * const.e)

    @property
    def num_photons(self) -> float:
        wavelength_m = self.wavelength_nm * 1e-9
        e_photon_joules = (const.h * const.c) / wavelength_m
        base_photons = self.pulse_energy_j / e_photon_joules
        gain = self.cavity_finesse_gain if self.enable_optical_cavity else 1.0
        return base_photons * gain

    @property
    def rayleigh_length_m(self) -> float:
        wavelength_m = self.wavelength_nm * 1e-9
        w0_m = self.waist_um * 1e-6
        return (np.pi * w0_m**2) / (wavelength_m * self.m2_factor)

    @property
    def fourier_bandwidth_pct(self) -> float:
        tau_s = self.pulse_duration_fs * 1e-15
        wavelength_m = self.wavelength_nm * 1e-9
        return (wavelength_m / (np.pi * const.c * tau_s)) * 100.0

    @property
    def a0(self) -> float:
        w0_m = self.waist_um * 1e-6
        tau_s = self.pulse_duration_fs * 1e-15
        effective_energy = self.pulse_energy_j * (self.cavity_finesse_gain if self.enable_optical_cavity else 1.0)
        peak_power = effective_energy / (tau_s * np.sqrt(np.pi / (4.0 * np.log(2.0))))
        peak_intensity = (2.0 * peak_power) / (np.pi * (w0_m * self.m2_factor)**2)
        lambda_um = self.wavelength_nm / 1000.0
        intensity_18 = (peak_intensity / 1e4) / 1e18
        return 0.855 * lambda_um * np.sqrt(intensity_18)


class MasterpieceICSModel:
    def __init__(self, e_beam: MasterpieceElectronBeam, laser: MasterpieceLaser):
        self.e_beam = e_beam
        self.laser = laser

    def optimize_collimation_angle(self) -> float:
        pol_factor = 1.0 if self.laser.polarization.lower() == "circular" else 0.5
        a0 = self.laser.a0
        gamma = self.e_beam.gamma
        theta_div = self.e_beam.divergence_rad
        
        bw_energy = 2.0 * self.e_beam.energy_spread_pct
        if self.laser.pulse_profile == "flat_top":
            bw_a0 = 0.5 * (pol_factor * a0**2) / (1.0 + pol_factor * a0**2) * 100.0
        else:
            bw_a0 = (pol_factor * a0**2) / (1.0 + pol_factor * a0**2) * 100.0
            
        bw_emittance = 0.5 * ((gamma * theta_div)**2 / (1.0 + pol_factor * a0**2)) * 100.0
        bw_ft = self.laser.fourier_bandwidth_pct
        
        bw_intrinsic = np.sqrt(bw_energy**2 + bw_a0**2 + bw_emittance**2 + bw_ft**2)
        target_bw_col = max(0.000001, bw_intrinsic)
        th_rad = np.sqrt((target_bw_col / 100.0) * 2.0 * (1.0 + pol_factor * a0**2)) / gamma
        return th_rad * 1e3

    def calculate_metrics(self) -> Dict[str, Any]:
        gamma = self.e_beam.gamma
        e_laser_ev = self.laser.photon_energy_ev
        a0 = self.laser.a0
        
        m_e_ev = (const.m_e * const.c**2) / const.e
        X = (4.0 * gamma * e_laser_ev) / m_e_ev
        kn_factor = klein_nishina_ratio(X)
        
        pol_factor = 1.0 if self.laser.polarization.lower() == "circular" else 0.5
        redshift_denom = 1.0 + pol_factor * (a0**2) + X
        
        phi_rad = np.radians(self.laser.collision_angle_deg)
        kinematic_factor = 2.0 * (gamma**2) * (1.0 - np.cos(phi_rad))
        e_peak_ev = (kinematic_factor * e_laser_ev) / redshift_denom
        
        sigma_x = self.e_beam.spot_size_um * 1e-6
        w0 = self.laser.waist_um * 1e-6 * self.laser.m2_factor
        eff_area = 2.0 * np.pi * (sigma_x**2 + (w0/2.0)**2)
        
        z_r = self.laser.rayleigh_length_m
        z_bunch = self.e_beam.bunch_length_m
        
        if self.e_beam.enable_beta_matching:
            hourglass_factor = 0.995
        else:
            hourglass_factor = 1.0 / np.sqrt(1.0 + (z_bunch / (2.0 * z_r))**2)

        sigma_t = 6.6524587321e-29  
        raw_yield = (
            (self.e_beam.num_electrons * self.laser.num_photons / eff_area)
            * sigma_t * kn_factor * hourglass_factor
        )
        
        optimal_theta_c_mrad = self.optimize_collimation_angle()
        theta_c_rad = optimal_theta_c_mrad * 1e-3
        theta_div = self.e_beam.divergence_rad
        
        theta_xray = np.sqrt(((1.0 + pol_factor * a0**2) / (gamma**2)) + theta_div**2)
        pinhole_theta = self.laser.pinhole_radius_mm / (self.laser.pinhole_distance_m * 1e3)
        effective_acceptance = min(theta_c_rad, pinhole_theta)
        
        aperture_eff = min(1.0, (effective_acceptance / max(1e-6, theta_xray))**2)
        
        bw_collimation = 0.5 * ((gamma * effective_acceptance)**2 / (1.0 + pol_factor * a0**2)) * 100.0
        bw_energy = 2.0 * self.e_beam.energy_spread_pct
        
        if self.laser.pulse_profile == "flat_top":
            bw_a0 = 0.5 * (pol_factor * a0**2) / (1.0 + pol_factor * a0**2) * 100.0
        else:
            bw_a0 = (pol_factor * a0**2) / (1.0 + pol_factor * a0**2) * 100.0
            
        bw_emittance = 0.5 * ((gamma * theta_div)**2 / (1.0 + pol_factor * a0**2)) * 100.0
        bw_laser_ft = self.laser.fourier_bandwidth_pct
        
        masterpiece_rms_bw = np.sqrt(bw_energy**2 + bw_a0**2 + bw_emittance**2 + bw_laser_ft**2 + bw_collimation**2)
        
        array_t = self.laser.channel_cut_transmission if self.laser.enable_channel_cut_array else 1.0
        masterpiece_total_yield = raw_yield * aperture_eff * array_t
        
        # Practical Jitter & Stability Robustness Factor
        timing_jitter_factor = np.exp(-0.5 * ((15.0 / self.e_beam.bunch_length_fs)**2))
        pointing_stability_factor = 1.0 - min(0.15, (optimal_theta_c_mrad * 0.05))
        practical_yield = masterpiece_total_yield * timing_jitter_factor * pointing_stability_factor
        
        return {
            "gamma": gamma,
            "a0": a0,
            "peak_energy_kev": e_peak_ev / 1000.0,
            "masterpiece_total_yield": masterpiece_total_yield,
            "practical_yield": practical_yield,
            "optimal_collimation_mrad": optimal_theta_c_mrad,
            "pinhole_angle_mrad": pinhole_theta * 1e3,
            "aperture_eff": aperture_eff,
            "bandwidth_rms_pct": masterpiece_rms_bw,
            "bandwidth_ev": (e_peak_ev * (masterpiece_rms_bw / 100.0)),
            "bw_energy_pct": bw_energy,
            "bw_a0_pct": bw_a0,
            "bw_emittance_pct": bw_emittance,
            "bw_laser_ft_pct": bw_laser_ft,
            "bw_collimation_pct": bw_collimation,
            "cavity_gain": self.laser.cavity_finesse_gain if self.laser.enable_optical_cavity else 1.0,
            "peak_current_A": self.e_beam.peak_current_A,
            "initial_emittance_um": self.e_beam.norm_emittance_um,
            "effective_emittance_um": self.e_beam.effective_norm_emittance_um,
            "array_transmission": array_t
        }

    def generate_spectrum(self, num_points: int = 5000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        metrics = self.calculate_metrics()
        peak_kev = metrics["peak_energy_kev"]
        sigma_final = max(1e-9, peak_kev * (metrics["bandwidth_rms_pct"] / 100.0))
        
        grid_half_width = 12.0 * sigma_final
        e_grid = np.linspace(max(0.1, peak_kev - grid_half_width), peak_kev + grid_half_width, num_points)
        
        spectrum = compute_masterpiece_spectrum_numba(
            e_grid,
            peak_kev,
            sigma_final,
            metrics["practical_yield"]
        )
        return e_grid, spectrum, metrics


def run_optuna_nsgaii_optimization() -> Tuple[MasterpieceElectronBeam, MasterpieceLaser, optuna.Study]:
    """Optuna Multi-Objective NSGA-II Optimization Engine with Study Return (v11.7)."""
    print("[INFO] Optuna Multi-Objective NSGA-II ධාවනය වෙමින් පවතී (v11.7 Research Paper Edition)...")
    
    def objective(trial: optuna.Trial):
        charge_pc = trial.suggest_float("charge_pc", 200.0, 800.0)
        pulse_energy_j = trial.suggest_float("pulse_energy_j", 0.02, 0.08)
        cavity_gain = trial.suggest_float("cavity_gain", 1500.0, 4000.0)
        pinhole_mm = trial.suggest_float("pinhole_radius_mm", 0.020, 0.045)
        waist_um = trial.suggest_float("waist_um", 100.0, 220.0)
        
        e_beam = MasterpieceElectronBeam(charge_pc=charge_pc)
        laser = MasterpieceLaser(
            pulse_energy_j=pulse_energy_j,
            cavity_finesse_gain=cavity_gain,
            pinhole_radius_mm=pinhole_mm,
            waist_um=waist_um
        )
        
        model = MasterpieceICSModel(e_beam, laser)
        metrics = model.calculate_metrics()
        
        y = metrics["practical_yield"]
        bw = metrics["bandwidth_rms_pct"]
        
        if np.isnan(bw) or np.isnan(y) or y <= 0:
            return 10.0, 0.0
            
        return bw, y

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    study.optimize(objective, n_trials=120, show_progress_bar=False)
    
    pareto_trials = study.best_trials if study.best_trials else study.trials
    valid_trials = [t for t in pareto_trials if t.values[0] <= 0.025]
    if valid_trials:
        best_trial = max(valid_trials, key=lambda t: t.values[1])
    else:
        best_trial = max(pareto_trials, key=lambda t: t.values[1] / (t.values[0] + 1e-9))
    
    best_params = best_trial.params
    best_e_beam = MasterpieceElectronBeam(charge_pc=best_params["charge_pc"])
    best_laser = MasterpieceLaser(
        pulse_energy_j=best_params["pulse_energy_j"],
        cavity_finesse_gain=best_params["cavity_gain"],
        pinhole_radius_mm=best_params["pinhole_radius_mm"],
        waist_um=best_params["waist_um"]
    )
    return best_e_beam, best_laser, study


def plot_masterpiece_results(
    energy_grid: np.ndarray,
    spectrum: np.ndarray,
    metrics: Dict[str, Any],
    output_filename: str = "ics_masterpiece_v11.7_spectrum_breakdown.png"
) -> None:
    plt.rcParams.update({'font.size': 9.5, 'font.family': 'sans-serif'})
    
    # 1. Main Spectrum & Component Breakdown Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11.0), dpi=300)
    
    ax1.plot(energy_grid, spectrum, color="#0072BD", lw=2.2, 
             label=f"Robust Ultra-Narrowband Spectrum (Array T = {metrics['array_transmission']*100:.0f}%)")
    ax1.axvline(metrics["peak_energy_kev"], color="#D9534F", linestyle="--", lw=1.6, 
                label=f"Fundamental Peak: {metrics['peak_energy_kev']:.4f} keV\nRMS Bandwidth: {metrics['bandwidth_ev']*1000.0:.3f} meV ({metrics['bandwidth_rms_pct']:.7f}%)\nPractical Yield (Jitter-Corrected): {metrics['practical_yield']:.2e} ph/shot")
    ax1.set_xlabel("Photon Energy (keV)", fontweight="bold")
    ax1.set_ylabel("Spectral Density (ph / keV / shot)", fontweight="bold")
    ax1.set_title("v11.7 Optuna NSGA-II Publication-Ready ICS Spectrum & Robustness Breakdown", fontweight="bold", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="#F9F9F9", fontsize=8.5)
    
    categories = ['Intrinsic Energy\nSpread', 'Non-linear a0\nBroadening', 'Space-Charge\nEmittance', 'Laser Fourier\nLimit', 'Collimation\nAperture']
    values = [metrics['bw_energy_pct'], metrics['bw_a0_pct'], metrics['bw_emittance_pct'], 
              metrics['bw_laser_ft_pct'], metrics['bw_collimation_pct']]
    colors = ['#2E7D32', '#C62828', '#6A1B9A', '#EF6C00', '#1565C0']
    
    bars = ax2.bar(categories, values, color=colors, width=0.45, alpha=0.88, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel("RMS Bandwidth Contribution (%)", fontweight="bold")
    ax2.set_title("RMS Bandwidth Component Breakdown (Robust Model)", fontweight="bold", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, axis='y')
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1e-9, f"{yval:.7f}%", ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] ප්‍රධාන ප්‍රස්තාරය සේව් විය: {output_filename}")
    open_image_file(output_filename)


def plot_optuna_pareto_front(study: optuna.Study, output_filename: str = "ics_masterpiece_v11.7_pareto_front.png") -> None:
    """Generates and saves the Optuna NSGA-II Pareto Front Optimization Plot."""
    plt.figure(figsize=(7, 5), dpi=300)
    
    bandwidths = []
    yields = []
    for t in study.trials:
        if t.values is not None and not np.isnan(t.values[0]) and not np.isnan(t.values[1]):
            bandwidths.append(t.values[0])
            yields.append(t.values[1])
            
    plt.scatter(bandwidths, yields, c='#0072BD', alpha=0.6, edgecolor='k', s=35, label='Optuna Trials')
    
    pareto_bw = [t.values[0] for t in study.best_trials if t.values is not None]
    pareto_y = [t.values[1] for t in study.best_trials if t.values is not None]
    plt.scatter(pareto_bw, pareto_y, c='#D9534F', s=90, marker='*', edgecolor='black', label='Pareto Optimal Front')
    
    plt.xlabel("RMS Bandwidth (%)", fontweight="bold")
    plt.ylabel("Practical Photon Yield (ph/shot)", fontweight="bold")
    plt.title("v11.7 Optuna NSGA-II Multi-Objective Pareto Front", fontweight="bold", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="#F9F9F9")
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Pareto Front ප්‍රස්තාරය සේව් විය: {output_filename}")
    open_image_file(output_filename)


def plot_parameter_sensitivity(output_filename: str = "ics_masterpiece_v11.7_sensitivity.png") -> None:
    """Generates and saves the Parameter Sensitivity Analysis Plot (Laser Waist vs Bandwidth & Yield)."""
    waist_range = np.linspace(100.0, 220.0, 35)
    bandwidths = []
    yields = []
    
    for w0 in waist_range:
        eb = MasterpieceElectronBeam()
        las = MasterpieceLaser(waist_um=w0)
        mod = MasterpieceICSModel(eb, las)
        m = mod.calculate_metrics()
        bandwidths.append(m['bandwidth_rms_pct'])
        yields.append(m['practical_yield'])
        
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    color = '#0072BD'
    ax1.set_xlabel('Laser Waist $w_0$ ($\mu$m)', fontweight='bold')
    ax1.set_ylabel('RMS Bandwidth (%)', color=color, fontweight='bold')
    ax1.plot(waist_range, bandwidths, color=color, lw=2.2, label='RMS Bandwidth')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    ax2 = ax1.twinx()  
    color = '#D9534F'
    ax2.set_ylabel('Practical Yield (ph/shot)', color=color, fontweight='bold')
    ax2.plot(waist_range, yields, color=color, lw=2.2, linestyle='--', label='Practical Yield')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('v11.7 Parameter Sensitivity: Laser Waist ($w_0$) vs Performance', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Sensitivity ප්‍රස්තාරය සේව් විය: {output_filename}")
    open_image_file(output_filename)


if __name__ == "__main__":
    e_beam, laser, study = run_optuna_nsgaii_optimization()
    
    model = MasterpieceICSModel(e_beam, laser)
    e_grid, spec, metrics = model.generate_spectrum()
    
    print("=================================================================================")
    print("      ULTIMATE MASTERPIECE OPTUNA NSGA-II ICS ENGINE v11.7 (PAPER EDITION)       ")
    print("=================================================================================")
    print(f"Peak X-Ray Energy                          : {metrics['peak_energy_kev']:.4f} keV")
    print(f"Jitter-Corrected Practical Photon Yield    : {metrics['practical_yield']:.4e} photons/shot")
    print(f"Final RMS Bandwidth                        : {metrics['bandwidth_rms_pct']:.7f} % ({metrics['bandwidth_ev']*1000.0:.4f} meV)")
    print("=================================================================================")
    
    # Generate all 3 publication-ready figures automatically
    plot_masterpiece_results(e_grid, spec, metrics, "ics_masterpiece_v11.7_spectrum_breakdown.png")
    plot_optuna_pareto_front(study, "ics_masterpiece_v11.7_pareto_front.png")
    plot_parameter_sensitivity("ics_masterpiece_v11.7_sensitivity.png")
