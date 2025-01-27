# %% Imports et blabla
import contrast
from PIL import Image
from scipy import ndimage
from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import multiprocessing
from cycler import cycler
import tqdm
import time
import cv2
import os
import regex as re
import velocity
import pickle
import faulthandler
import scipy.optimize as opt
from scipy.optimize import curve_fit
from skimage.restoration import unwrap_phase
from scipy.integrate import solve_ivp
from scipy import optimize

# from findpeaks import findpeaks

faulthandler.enable()

# matplotlib.use("Qt5Agg")

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")
# for dark theme
# plt.style.use("dark_background")
# plt.rcParams["figure.facecolor"] = "#00000080"
# plt.rcParams["axes.facecolor"] = "#00000080"
# plt.rcParams["savefig.facecolor"] = "#00000080"
# plt.rcParams['savefig.transparent'] = True
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Liberation Sans']
# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.size": 12
# })
# sshfs taladjidi@patriot.lkb.upmc.fr:/partages/EQ15B/LEON-15B /home/tangui/LEON
# path_leon = "/run/user/1000/gvfs/sftp:host=patriot.lkb.upmc.fr,user=taladjidi/partages/EQ15B/LEON-15B"
# path_leon = "/run/user/1000/gvfs/sftp:host=88.160.142.14,port=16384,user=aladjidi/home/aladjidi/Disk0/LEON"

# %% Paths and parameters

k0 = 2 * np.pi / 780e-9
L = 20e-2
d_slm = 8e-6
d_real = (
    3.45e-6 * 2
)  # times 2 beacuse of the lose of resolution for amplitude and phase reconstruction
d_def = 1.1e-6
Nx, Ny = 2464, 2056

Nx_def, Ny_def = 2048, 2048

GPU = False  # if True, use nvidia GPU for spectral analysis (need cupy package)
if GPU:
    import cupy as cp

# %% Functions


def Vortex2D(mn, y_val, x_val, xi) -> np.ndarray:
    y, x = mn
    # y, x = np.meshgrid(m, n)
    # x = x - n / 2.0 + x_val
    # y = y - m / 2.0 + y_val
    r = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
    Psi = r / np.sqrt(r**2 + (2 * xi) ** 2)
    Amp = np.abs(Psi)
    return Amp.ravel()


def Soliton1D(x, Mach, xi):
    Psi = (
        np.sqrt(1 - Mach**2) * np.tanh((x / (np.sqrt(2) * xi)) * np.sqrt(1 - Mach**2))
        + 1j * Mach
    )
    # Amp = np.abs(Psi)
    Phase = np.angle(Psi) + np.pi
    return Phase


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def compute_phase_time(scan: str, plot: bool = False):
    """Compute the phase.

    Computes the phase from the interferograms for a time scan
    dataset.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the phase. Defaults to False.
    """
    # powers = np.load(f"{path_real}/{scan}/powers.npy")
    # N_times = powers.size
    input_power = np.load(f"{scan}/input_power.npy")
    N_times = input_power.size
    N_avg = np.load(f"{scan}/N_avg.npy")
    laser_powers = np.load(f"{scan}/laser_settings.npy")
    field = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    field_ref = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    field_vortex = np.zeros((N_times, N_avg, Ny // 2, Nx // 2), dtype=np.complex64)
    # ref_arm = np.array(Image.open(
    #     f"{path_real}/ref.TIF"), dtype=np.uint16).astype(np.float32)
    # ref_arm = ndimage.zoom(ref_arm, .5)
    taus = np.zeros(N_times)
    taus_err = np.zeros(N_times)
    # prepare plans
    a = pyfftw.empty_aligned((Ny, Nx), dtype=np.float32)
    c = pyfftw.empty_aligned((Ny // 2, Nx // 2), dtype=np.complex64)
    plan_fft = pyfftw.builders.rfft2(a)
    plan_ifft = pyfftw.builders.ifft2(c)
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    plans = (plan_fft, plan_ifft)
    # del a, b
    pbar = tqdm.tqdm(total=N_times * N_avg, desc="Computing phase", position=1)
    for i in range(N_times):
        tau = np.zeros(N_avg)
        for j in range(N_avg):
            im_ref = np.array(
                Image.open(f"{scan}/dn_ref_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            im = np.array(
                Image.open(f"{scan}/dn_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            im_vortex = np.array(
                Image.open(f"{scan}/dn_vortex_{i}_{j}.tiff"),
                dtype=np.float32,
            )
            field_ref[i, j, :, :] = contrast.im_osc_fast_t(
                im_ref,
                plans=plans,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 16,
            )
            field[i, j, :, :] = contrast.im_osc_fast_t(
                im,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 16,
            )
            field_vortex[i, j, :, :] = contrast.im_osc_fast_t(
                im_vortex,
                plans=plans,
                center=(im_ref.shape[-2] // 2, im_ref.shape[-1] // 4),
                radius=im_ref.shape[-2] // 16,
            )
            # plt.imshow(np.abs(field[i, j, :, :]) ** 2)
            # plt.show()
            phi_ref = contrast.angle_fast(field_ref[i, j, :, :])
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            threshold = 2e-2
            mask = rho_ref < threshold * np.max(rho_ref)
            phi_ref_masked = np.ma.array(phi_ref, mask=mask)
            phi_ref_unwrapped = restoration.unwrap_phase(
                phi_ref_masked, wrap_around=(True, True)
            )
            tau[j] = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))
            pbar.update(1)
        taus[i] = np.mean(tau)
        taus_err[i] = np.std(tau)
    pbar.close()
    print("Saving data ...")
    t0 = time.perf_counter()
    np.save(f"{scan}/field_ref.npy", field_ref)
    np.save(f"{scan}/field.npy", field)
    np.save(f"{scan}/field_vortex.npy", field_vortex)
    np.save(f"{scan}/taus_reproc.npy", taus)
    np.save(f"{scan}/taus_err_reproc.npy", taus_err)
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes + taus.nbytes
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.abs(field[0, -1, :, :]) ** 2)
    ax[1].imshow(np.angle(field[0, -1, :, :]), cmap="twilight_shifted")
    fig.savefig(f"{scan}/field.svg", dpi=300)
    taus0 = np.load(f"{scan}/taus.npy")
    taus0_err = np.load(f"{scan}/taus_err.npy")
    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        laser_powers,
        taus0,
        yerr=taus0_err,
        marker="o",
        label="During acquisition",
    )
    ax1.errorbar(laser_powers, taus, yerr=taus_err, marker="o", label="Reprocessed")
    ax1.legend()
    ax1.set_xlabel("Power in a.u.")
    ax1.set_ylabel(r"$\delta\phi = \tau$ in rad")
    ax1.set_title("Phase shift vs power")
    fig1.savefig(f"{scan}/taus.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def compute_n2_Isat(scan: str, plot: bool = False):
    """Compute n2 and Isat from the phase shift vs power and plot the effective refractive index"""
    print("Loading data ...")

    def delta_n_avg(I, I_sat, n2):
        return n2 * I

    waist = 1.7e-3
    L = 0.2
    k0 = 2 * np.pi / 780e-9
    taus = np.load(f"{scan}/taus.npy")
    delta_n = taus / (k0 * L)
    # laser_powers = np.load(f"{scan}/laser_settings.npy")
    laser_powers = np.linspace(0.05, 3, len(delta_n))
    laser_intensity = 2 * laser_powers / (np.pi * (waist) ** 2)
    laser_intensity = laser_intensity[5:]
    delta_n = delta_n[5:]
    popt, pcov = curve_fit(delta_n_avg, laser_intensity, delta_n, p0=[6e5, 1e-10])
    plt.figure()
    plt.plot(laser_intensity, delta_n, marker="o")
    plt.plot(
        laser_intensity,
        delta_n_avg(laser_intensity, *popt),
        label=f"Isat = {popt[0]:.2e} W/m$^2$ | n2 = {popt[1]:.2e} m$^2$/W",
    )
    plt.xlabel("Laser intensity (W/m$^2$)")
    plt.ylabel(r"$\Delta n$")
    plt.legend()
    plt.grid()
    plt.savefig(f"{scan}/n2_Isat.svg", dpi=300)
    plt.show()


def compute_final_xi(scan: str, plot: bool = False):
    """Find final xi value"""
    print("Loading data ...")
    t0 = time.perf_counter()
    fields_ref = np.load(f"{scan}/field_ref.npy")
    fields = np.load(f"{scan}/field.npy")
    fields_vortex = np.load(f"{scan}/field_vortex.npy")
    taus = np.load(f"{scan}/taus.npy")
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    window_m = 30
    vort_size = []
    vort_size_err = []
    xi_avg = []
    for i in tqdm.tqdm(range(fields.shape[0]), desc="Plotting", position=1):
        vort_size_avg = []
        for j in range(fields.shape[1]):
            field = fields[i, j, :, :]
            field_ref = fields_ref[i, j, :, :]
            field_vortex = fields_vortex[i, j, :, :]
            field = ndimage.gaussian_filter(field, 2)
            field_ref = ndimage.gaussian_filter(field_ref, 2)
            field_vortex = ndimage.gaussian_filter(field_vortex, 2)
            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            contrast.exp_angle_fast(field_vortex, field_ref)
            phi_vort_flat = contrast.angle_fast(field_vortex)
            rho = field.real * field.real + field.imag * field.imag
            # rho = ndimage.gaussian_filter(rho, 3)
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho_vort = (
                field_vortex.real * field_vortex.real
                + field_vortex.imag * field_vortex.imag
            )
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)
            rho_vort = np.sqrt(rho_vort)
            rho_vort = rho_vort / np.nanmax(rho_vort)

            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi)
            window = int(radius_mask_ref) / 1.5
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = np.abs(
                (x - mask_ref.shape[-1] // 2) ** 2 + (y - mask_ref.shape[-2] // 2) ** 2
                < radius_mask_ref**2
            )
            mask_fluid = np.logical_not(mask_fluid)
            rho_masked = rho
            rho_masked[mask_fluid] = np.nan
            # plt.figure()
            # plt.imshow(rho_masked, cmap='gray')
            # plt.show()

            ####### Compute xi with vortex radius
            vortices = np.array([])
            vort = velocity.vortex_detection(phi_vort_flat, plot=False, r=1)
            vort[:, 0] -= phi_vort_flat.shape[-1] // 2
            vort[:, 1] -= phi_vort_flat.shape[-2] // 2
            # vort_in_window_x = np.logical_and(vort[:, 0] > -window, vort[:, 0] < window)
            # vort_in_window_y = np.logical_and(vort[:, 1] > -window, vort[:, 1] < window)
            # vort_select = np.logical_and(vort_in_window_x, vort_in_window_y)
            vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > window**2)
            vortices = vort[vort_select, :]
            center_x = field.shape[1] // 2 + vortices[0][0]
            center_y = field.shape[0] // 2 + vortices[0][1]
            # Find the vortex position, and width
            rho_zoom = rho_vort[
                int(center_y - window_m) : int(center_y + window_m),
                int(center_x - window_m) : int(center_x + window_m),
            ]
            x = np.arange(rho_zoom.shape[1])
            y = np.arange(rho_zoom.shape[0])
            X, Y = np.meshgrid(x, y)
            inital_guess = (window_m, window_m, 10)
            try:
                popt, pcov = opt.curve_fit(
                    Vortex2D, (Y, X), rho_zoom.ravel(), p0=inital_guess
                )
            except RuntimeError:
                popt = np.array([window_m, window_m, np.nan])
            vortex_reshape = Vortex2D((Y, X), *popt).reshape(window_m * 2, window_m * 2)
            # plt.figure()
            # plt.imshow(vortex_reshape, cmap='inferno')
            # plt.show()
            vort_rad = popt[2] * d_real
            if vort_rad > 150e-6:
                vort_rad = np.nan
            vort_size_avg += [vort_rad]
        vort_size_avg = np.array(vort_size_avg)
        vort_size += [np.nanmean(vort_size_avg)]
        vort_size_err += [np.nanstd(vort_size_avg)]
        xi_avg += [np.sqrt(0.2 / (taus[i] * 8e6))]
    vort_size = np.array(vort_size)
    vort_size_err = np.array(vort_size_err)
    xi_avg = np.array(xi_avg)
    np.save(f"{scan}/xis_final.npy", vort_size)
    plt.figure()
    plt.errorbar(
        taus,
        vort_size * 1e6,
        yerr=vort_size_err * 1e6,
        marker="o",
        label="Vortex radius",
    )
    # plt.plot(taus, vort_size*1e6, marker='o', label='Vortex radius')
    plt.plot(taus, xi_avg * 1e6, marker="s", label="Average xi")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\xi$ $(\mu m)$")
    plt.grid()
    plt.legend()
    plt.savefig(f"{scan}/xi_vortex.svg", dpi=300)
    if plot:
        plt.show()


def plot_fields(scan: str, window: int = 50, plot: bool = False):
    """Plot the fields.

    Args:
        scan (str): Path to the dataset
        plot (bool, optional): Whether to plot the fields. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{scan}/taus.npy")
    field_ref = np.load(f"{scan}/field_ref.npy")
    field = np.load(f"{scan}/field.npy")
    t = time.perf_counter() - t0
    sz = field_ref.nbytes + field.nbytes
    field = field[:, -1, :, :]
    field_ref = field_ref[:, -1, :, :]
    rate = sz / t
    # if field_ref.shape[-1] == Nx // 2:
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    delta_n = taus / (k0 * L)
    # xis = 1 / (k0 * np.sqrt(delta_n))
    xis = np.load(f"{scan}/xis_final.npy")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    im0 = ax[0].imshow(
        np.ones(field.shape[1:]),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    im1 = ax[1].imshow(
        np.ones(field.shape[1:]),
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="none",
    )
    for a in ax:
        a.set_xlabel(r"$x/\xi$")
        a.set_ylabel(r"$y/\xi$")
    ax[0].set_title("Density")
    ax[1].set_title("Normalized phase")
    # ax[0].locator_params(axis="both", nbins=5)
    # ax[1].locator_params(axis="both", nbins=5)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    fig.colorbar(im0, ax=ax[0], label=r"$\sqrt{\rho}$ in a.u.", shrink=0.6)
    cbar = fig.colorbar(im1, ax=ax[1], label=r"$\phi/\phi_0$ in rad", shrink=0.6)
    cbar.set_ticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[
            r"$\pi$",
            r"$-\frac{\pi}{2}$",
            r"$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ],
    )
    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    # create folder if it does not exist
    if not (os.path.exists(f"{scan}")):
        os.mkdir(f"{scan}")
    video_writer = cv2.VideoWriter(
        f"{scan}/field.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    video_writer1 = cv2.VideoWriter(
        f"{scan}/field_rescaled.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        3,
        (mat.shape[1], mat.shape[0]),
    )
    for i in tqdm.tqdm(range(field.shape[0]), desc="Plotting fields", position=1):
        fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xis[i] * 1e6:.0f} $\mu m$")
        ext_real = [
            -field.shape[-1] * d_real / (2 * xis[i]),
            field.shape[-1] * d_real / (2 * xis[i]),
            -field.shape[-2] * d_real / (2 * xis[i]),
            field.shape[-2] * d_real / (2 * xis[i]),
        ]
        contrast.exp_angle_fast(field[i, :, :], field_ref[i, :, :])
        phi_flat = contrast.angle_fast(field[i, :, :])
        rho = (
            field[i, :, :].real * field[i, :, :].real
            + field[i, :, :].imag * field[i, :, :].imag
        )
        rho = np.sqrt(rho)
        rho = rho / np.max(rho)
        im0.set_data(rho)
        im0.set_clim(np.min(rho), 1.0)
        im1.set_data(phi_flat)
        im0.set_extent(ext_real)
        im1.set_extent(ext_real)
        for a in ax:
            a.set_xlim((ext_real[0], ext_real[1]))
            a.set_ylim((ext_real[2], ext_real[3]))
        fig.canvas.draw()
        if plot:
            plt.show()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer.write(mat)
        fig.savefig(f"{scan}/fields_{i}.svg", dpi=300)
        window = window
        for a in ax:
            a.set_xlim((-window, window))
            a.set_ylim((-window, window))
        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        video_writer1.write(mat)
        fig.savefig(f"{scan}/fields_rescaled_{i}.svg", dpi=300)
        if plot:
            plt.show()
    print()
    plt.close("all")
    video_writer.release()
    video_writer1.release()


def energy_time(scan: str, plot: bool = False):
    """Compute the kinetic energy (comp and inc) and the interaction energy.

    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    field_ref = np.load(f"{scan}/field_ref.npy")

    N_times = field_ref.shape[0]
    N_avg = field_ref.shape[1]
    field = np.load(f"{scan}/field.npy")
    field_vortex = np.load(f"{scan}/field_vortex.npy")
    taus = np.load(f"{scan}/taus.npy")
    taus_err = np.load(f"{scan}/taus_err.npy")
    t = time.perf_counter() - t0

    sz = field_ref.nbytes + field.nbytes + field_vortex.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    filtering_radius = 4
    # field = ndimage.gaussian_filter(field, filtering_radius)
    # field_vortex = ndimage.gaussian_filter(field_vortex, filtering_radius)
    # compute energies
    e_comp = np.zeros(field.shape[:2])
    e_inc = np.zeros(field.shape[:2])
    e_int = np.zeros(field.shape[:2])
    u_inc = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    e_comp_vortex = np.zeros(field.shape[:2])
    e_inc_vortex = np.zeros(field.shape[:2])
    e_int_vortex = np.zeros(field.shape[:2])
    u_inc_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    u_comp_vortex = np.zeros(
        (*field.shape[:2], 2, field.shape[-2], field.shape[-1]),
        dtype=np.float32,
    )
    for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
        # for i in range(30,31):
        for j in range(N_avg):
            contrast.exp_angle_fast(field[i, j, :, :], field_ref[i, j, :, :])
            contrast.exp_angle_fast_scalar(
                field[i, j, :, :],
                field[i, j, field.shape[-2] // 2, field.shape[-1] // 2],
            )
            contrast.exp_angle_fast(field_vortex[i, j, :, :], field_ref[i, j, :, :])
            contrast.exp_angle_fast_scalar(
                field_vortex[i, j, :, :],
                field_vortex[
                    i,
                    j,
                    field_vortex.shape[-2] // 2,
                    field_vortex.shape[-1] // 2,
                ],
            )
            rho = (
                field[i, j, :, :].real * field[i, j, :, :].real
                + field[i, j, :, :].imag * field[i, j, :, :].imag
            )
            rho_vortex = (
                field_vortex[i, j, :, :].real * field_vortex[i, j, :, :].real
                + field_vortex[i, j, :, :].imag * field_vortex[i, j, :, :].imag
            )
            rho_ref = (
                field_ref[i, j, :, :].real * field_ref[i, j, :, :].real
                + field_ref[i, j, :, :].imag * field_ref[i, j, :, :].imag
            )
            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi) / 1.6
            x, y = np.meshgrid(
                np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2])
            )
            mask_fluid = (x - mask_ref.shape[-1] // 2) ** 2 + (
                y - mask_ref.shape[-2] // 2
            ) ** 2 < radius_mask_ref**2
            mask_fluid = np.logical_not(mask_fluid)

            field[i, j, :, :] = ndimage.gaussian_filter(
                field[i, j, :, :], filtering_radius
            )
            (
                _,
                u_inc[i, j, :, :, :],
                u_comp[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(field[i, j, :, :], dx=d_real, plot=False)
            # u_inc[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_inc[i, j, :, :, :], filtering_radius
            # )
            # u_comp[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_comp[i, j, :, :, :], filtering_radius
            # )
            for k in range(u_comp_vortex.shape[2]):
                u_comp[i, j, k, :, :][mask_fluid] = 0
                u_inc[i, j, k, :, :][mask_fluid] = 0
            e_comp[i, j] = np.sum(u_comp[i, j, :, :, :] ** 2)
            e_inc[i, j] = np.sum(u_inc[i, j, :, :, :] ** 2)
            e_int[i, j] = np.sum(rho * rho)

            field_vortex[i, j, :, :] = ndimage.gaussian_filter(
                field_vortex[i, j, :, :], filtering_radius
            )
            (
                _,
                u_inc_vortex[i, j, :, :, :],
                u_comp_vortex[i, j, :, :, :],
            ) = velocity.helmholtz_decomp(
                field_vortex[i, j, :, :], dx=d_real, plot=False
            )
            # u_inc_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_inc_vortex[i, j, :, :, :], filtering_radius
            # )
            # u_comp_vortex[i, j, :, :, :] = ndimage.gaussian_filter(
            #     u_comp_vortex[i, j, :, :, :], filtering_radius
            # # )
            for jj in range(u_comp_vortex.shape[2]):
                u_comp_vortex[i, j, k, :, :][mask_fluid] = 0
                u_inc_vortex[i, j, k, :, :][mask_fluid] = 0
            e_comp_vortex[i, j] = np.sum(np.abs(u_comp_vortex[i, j, :, :, :]) ** 2)
            e_inc_vortex[i, j] = np.sum(np.abs(u_inc_vortex[i, j, :, :, :]) ** 2)
            e_int_vortex[i, j] = np.sum(rho_vortex * rho_vortex)
    print("\nSaving data ...")
    t0 = time.perf_counter()
    # np.save(f"{path_dn}/{scan}/e_comp.npy", e_comp)
    # np.save(f"{path_dn}/{scan}/e_inc.npy", e_inc)
    # np.save(f"{path_dn}/{scan}/e_int.npy", e_int)
    np.save(f"{scan}/u_inc.npy", u_inc)
    # np.save(f"{path_dn}/{scan}/u_comp.npy", u_comp)
    # np.save(f"{path_dn}/{scan}/e_comp_vortex.npy", e_comp_vortex)
    # np.save(f"{path_dn}/{scan}/e_inc_vortex.npy", e_inc_vortex)
    # np.save(f"{path_dn}/{scan}/e_int_vortex.npy", e_int_vortex)
    # np.save(f"{path_dn}/{scan}/u_inc_vortex.npy", u_inc_vortex)
    # np.save(f"{path_dn}/{scan}/u_comp_vortex.npy", u_comp_vortex)
    t = time.perf_counter() - t0
    sz = (
        e_comp.nbytes
        + e_inc.nbytes
        + u_inc.nbytes
        + u_comp.nbytes
        + e_comp_vortex.nbytes
        + e_inc_vortex.nbytes
        + u_inc_vortex.nbytes
        + u_comp_vortex.nbytes
    )
    rate = sz / t
    print(f"Saved {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title(r"$E_{two}$")
    ax[0].errorbar(
        taus,
        np.mean(e_comp, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[0].errorbar(
        taus,
        np.mean(e_inc, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc, axis=-1),
        label="Incompressible",
        marker="^",
    )
    # ax[0].errorbar(
    #     taus,
    #     np.mean(e_int, axis=-1),
    #     xerr=taus_err,
    #     yerr=np.std(e_int, axis=-1),
    #     label="Interaction",
    #     marker="^",
    # )
    ax[1].set_title(r"$E_{one}$")
    ax[1].errorbar(
        taus,
        np.mean(e_comp_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_comp_vortex, axis=-1),
        label="Compressible",
        marker="o",
    )
    ax[1].errorbar(
        taus,
        np.mean(e_inc_vortex, axis=-1),
        xerr=taus_err,
        yerr=np.std(e_inc_vortex, axis=-1),
        label="Incompressible",
        marker="^",
    )
    # ax[1].errorbar(
    #     taus,
    #     np.mean(e_int_vortex, axis=-1),
    #     xerr=taus_err,
    #     yerr=np.std(e_int_vortex, axis=-1),
    #     label="Interaction",
    #     marker="^",
    # )
    ax[2].set_title(r"$E_{two}/E_{one}$")
    ax[2].errorbar(
        taus,
        np.mean(e_comp, axis=-1) / np.mean(e_comp_vortex, axis=-1),
        yerr=np.std(e_comp / e_comp_vortex, axis=-1),
        xerr=taus_err,
        label="Compressible",
        marker="o",
    )
    ax[2].errorbar(
        taus,
        np.mean(e_inc, axis=-1) / np.mean(e_inc_vortex, axis=-1),
        yerr=np.std(e_inc / e_inc_vortex, axis=-1),
        xerr=taus_err,
        label="Incompressible",
        marker="^",
    )
    ax[2].errorbar(
        taus,
        np.mean(e_int, axis=-1) / np.mean(e_int_vortex, axis=-1),
        yerr=np.std(e_int / e_int_vortex, axis=-1),
        xerr=taus_err,
        label="Interaction",
        marker="s",
    )
    for a in ax:
        a.set_xlabel(r"$\tau$ in rad")
        a.set_ylabel(r"$E$ in a.u.")
        a.legend()
    ax[2].axhline(1, color="grey", linestyle="--")
    ax[2].set_ylim(bottom=-0.05)
    fig.savefig(f"{scan}/energies.svg", dpi=300)
    if plot:
        plt.show()
    plt.close("all")


def spectral_analysis(
    scan: str,
    which: str = "vortex",
    img_nbr: int = 1,
    plot: bool = False,
    debug: bool = False,
):
    """Compute the energy spectra for a given time with the new spectral analysis."""
    print("Loading data ...")
    xi = 40e-6
    im = Image.open(f"{scan}/ref/fluid.tif")
    im_ref = Image.open(f"{scan}/ref/ref.tif")
    fluid = np.array(im)
    ref = np.array(im_ref)

    ## Choose image size
    # h, w = (206, 206)
    # h, w = (512, 512)
    h, w = (1024, 1024)

    e_inc_fields = []
    e_comp_fields = []
    g_inc_fields = []
    g_comp_fields = []
    filtering_radius = 2  # If no smooth set to 0
    exponent = 8  # Hypergaussian for beam edge smoothing
    scale = 4  # Oversampling factor

    field_ref = contrast.im_osc_fast_t(
        fluid / ref,
        center=(fluid.shape[-2] // 2, fluid.shape[-1] // 4),
        radius=fluid.shape[-2] // 8,
    )
    rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag

    # Define a circular mask from the reference field
    threshold = 5e-2
    mask_ref = rho_ref < threshold * np.max(rho_ref)
    mask_fluid = rho_ref >= threshold * np.max(rho_ref)
    radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi) / 1.05
    x, y = np.meshgrid(np.arange(mask_ref.shape[-1]), np.arange(mask_ref.shape[-2]))
    x = x - mask_ref.shape[-1] // 2
    y = y - mask_ref.shape[-2] // 2
    ## Hypergaussian
    E0 = np.exp(-(((x**2 + y**2) / radius_mask_ref**2) ** exponent))

    for i in np.arange(1, img_nbr + 1):
        dn = Image.open(f"{scan}/{which}/{which}_{i}.tif")
        im_field = np.array(dn)
        field = contrast.im_osc_fast_t(
            im_field / ref,
            center=(im_field.shape[-2] // 2, im_field.shape[-1] // 4),
            radius=im_field.shape[-2] // 8,
        )
        contrast.exp_angle_fast(field, field_ref)
        field = ndimage.gaussian_filter(field, filtering_radius)
        center = (field.shape[-2] // 2, field.shape[-1] // 2)
        roi = (
            slice(center[0] - h // 2, center[0] + h // 2),
            slice(center[1] - w // 2, center[1] + w // 2),
        )
        field = field * np.abs(E0)  # Smooth edges
        field = field[roi]
        if debug:
            figure, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.abs(field), cmap="gray", interpolation="none")
            ax[1].imshow(np.angle(field), cmap="twilight_shifted", interpolation="none")
            plt.show()
        ks = 2 * np.pi * np.fft.fftfreq(field.shape[-1] * scale, d=d_real)
        ks = ks[0 : field.shape[0] // 2]
        rs = np.linspace(0, field.shape[0] // 2 * d_real * scale, ks.size)

        # if no nvidia GPU set cp=False
        g_inc_field, g_comp_field = velocity.corr_spectra(
            k=ks, r=rs, psi=field, d=d_real, cp=GPU
        )
        e_inc_field, e_comp_field = velocity.comp_incomp_spectrum(
            k=ks, psi=field, d=d_real, cp=GPU
        )

        e_inc_fields += [e_inc_field]
        e_comp_fields += [e_comp_field]
        g_inc_fields += [g_inc_field]
        g_comp_fields += [g_comp_field]
    e_inc_fields = np.array(e_inc_fields)
    e_inc_fields_mean = np.nanmean(e_inc_fields, axis=0)
    e_inc_fields_err = np.nanstd(e_inc_fields, axis=0)
    e_comp_fields = np.array(e_comp_fields)
    e_comp_fields_mean = np.nanmean(e_comp_fields, axis=0)
    e_comp_fields_err = np.nanstd(e_comp_fields, axis=0)
    g_inc_fields = np.array(g_inc_fields)
    g_inc_fields_mean = np.nanmean(g_inc_fields, axis=0)
    g_inc_fields_err = np.nanstd(g_inc_fields, axis=0)
    g_comp_fields = np.array(g_comp_fields)
    g_comp_fields_mean = np.nanmean(g_comp_fields, axis=0)
    g_comp_fields_err = np.nanstd(g_comp_fields, axis=0)

    R = radius_mask_ref * d_real
    Rx = R / xi

    if plot:
        rx = rs / xi
        plt.figure(figsize=(2, 5))
        plt.errorbar(
            rx,
            g_comp_fields_mean,
            yerr=g_comp_fields_err,
            marker="o",
            label="Compressible",
        )
        plt.errorbar(
            rx,
            g_inc_fields_mean,
            yerr=g_inc_fields_err,
            marker="^",
            label="Incompressible",
        )
        plt.xlabel(r"$r/\xi$")
        plt.ylabel(r"$g^{i}_{kin}$")
        plt.xlim(-2, 80)
        plt.axvline(Rx)
        plt.grid()
        plt.legend()
        plt.savefig(f"{scan}/{which}/g.svg", dpi=300)
        plt.show()
        np.save(f"{scan}/{which}/g_inc_fields_mean.npy", g_inc_fields_mean)
        np.save(f"{scan}/{which}/g_inc_fields_err.npy", g_inc_fields_err)
        np.save(f"{scan}/{which}/g_comp_fields_mean.npy", g_comp_fields_mean)
        np.save(f"{scan}/{which}/g_comp_fields_err.npy", g_comp_fields_err)
        np.save(f"{scan}/{which}/rx.npy", rx)

    if plot:
        kx = ks * xi
        plt.figure(figsize=(5, 3))
        plt.errorbar(
            kx,
            e_comp_fields_mean,
            yerr=e_comp_fields_err,
            marker="o",
            label="Compressible",
        )
        plt.errorbar(
            kx,
            e_inc_fields_mean,
            yerr=e_inc_fields_err,
            marker="^",
            label="Incompressible",
        )
        plt.xlabel(r"$k\xi$")
        plt.ylabel(r"$E^{c,i}_{kin}$")
        plt.yscale("log")
        plt.xscale("log")
        plt.tick_params(axis="both", direction="in", which="major", labelsize=13)
        plt.tick_params(axis="both", direction="in", which="minor", labelsize=13)
        plt.legend()
        plt.axvline(1, color="grey", linestyle="-")
        plt.axvline(2 * np.pi / Rx)
        plt.grid()
        plt.savefig(f"{scan}/{which}/e.svg", dpi=300)
        plt.show()
        np.save(f"{scan}/{which}/e_inc_fields_mean.npy", e_inc_fields_mean)
        np.save(f"{scan}/{which}/e_comp_fields_mean.npy", e_comp_fields_mean)
        np.save(f"{scan}/{which}/e_inc_fields_err.npy", e_inc_fields_err)
        np.save(f"{scan}/{which}/e_comp_fields_err.npy", e_comp_fields_err)
        np.save(f"{scan}/{which}/kx.npy", kx)


def normalized_velocity(scan: str, plot: bool = False):
    """Find the position of the quasi-soliton and compute its normalized energy
    by fitting the phase profile with the 1D soliton solution.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    fields_ref = np.load(f"{scan}/field_ref.npy")
    fields = np.load(f"{scan}/field.npy")
    fields_vortex = np.load(f"{scan}/field_vortex.npy")
    u_inc = np.load(f"{scan}/u_inc.npy")
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis_final.npy")
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    expo = 2
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")

    qs_velo = []
    qs_velo_err = []
    marker = []
    for i in tqdm.tqdm(range(fields.shape[0]), desc="Plotting", position=1):
        marker_s = 0
        qs_velo_avg = []
        for j in range(fields.shape[1]):
            field = fields[i, j, :, :]
            if 0 in fields[i, j, :, :]:
                field = fields[i, j - 1, :, :]
            field_ref = fields_ref[i, j, :, :]
            if 0 in fields_ref[i, j, :, :]:
                field_ref = fields_ref[i, j - 1, :, :]
            field_vortex = fields_vortex[i, j, :, :]
            if 0 in fields_vortex[i, j, :, :]:
                field_vortex = fields_vortex[i, j - 1, :, :]
            field = ndimage.gaussian_filter(field, expo)
            field_ref = ndimage.gaussian_filter(field_ref, expo)
            field_vortex = ndimage.gaussian_filter(field_vortex, expo)
            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            contrast.exp_angle_fast(field_vortex, field_ref)
            phi_jrs_flat = contrast.angle_fast(field)
            # phi_jrs_flat = ndimage.gaussian_filter(phi_jrs_flat, 3)
            phi_vort_flat = contrast.angle_fast(field_vortex)
            rho = field.real * field.real + field.imag * field.imag
            # rho = ndimage.gaussian_filter(rho, 3)
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho_vort = (
                field_vortex.real * field_vortex.real
                + field_vortex.imag * field_vortex.imag
            )
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)
            rho_vort = np.sqrt(rho_vort)
            rho_vort = rho_vort / np.nanmax(rho_vort)

            v = velocity.helmholtz_decomp(field, dx=d_real, plot=False)
            v_tot = np.gradient(unwrap_phase(phi_jrs_flat))
            v_tot_map = np.hypot(v_tot[0], v_tot[1])

            # Define a circular mask from the reference field
            threshold = 5e-2
            mask_ref = rho_ref < threshold * np.max(rho_ref)
            mask_fluid = rho_ref >= threshold * np.max(rho_ref)
            radius_mask_ref = np.sqrt(np.sum(mask_fluid) / np.pi)
            window = int(radius_mask_ref) / 1.5

            # Find qs position
            inc = (np.abs(u_inc[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
            inc /= np.nanmax(inc)
            # plt.figure()
            # plt.imshow(inc, cmap='gray')
            # plt.show()
            vortices = np.unravel_index(inc.argmax(), inc.shape)
            center_x = vortices[1]
            center_y = vortices[0]
            # print('\n', center_x, center_y, '\n')
            # End qs position

            w = 7.5
            # xi = np.sqrt(0.2/(taus[i]*8e6))
            xi = xis[i]
            ww = int(w * xi / d_real)
            size_norm = d_real / xi
            rho_zoom0 = rho[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            phi_zoom0 = phi_jrs_flat[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            vmap_zoom0 = v_tot_map[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            rho_ref_zoom0 = rho_ref[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            center = (rho_zoom0.shape[0] // 2, rho_zoom0.shape[0] // 2)

            # rho_zoom0 = np.abs(rho_zoom0)**2
            # rho_ref_zoom0 = np.abs(rho_ref_zoom0)**2
            # drho = rho_zoom0 - rho_ref_zoom0 + 1
            drho = np.abs(rho_zoom0)
            # drho = (rho_ref_zoom0 - rho_zoom0)
            # drho[drho < 0.4] = 0

            vortices = np.array([])
            vort = velocity.vortex_detection(phi_jrs_flat, plot=False, r=1)
            vort[:, 0] -= phi_jrs_flat.shape[-1] // 2
            vort[:, 1] -= phi_jrs_flat.shape[-2] // 2
            vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > window**2)
            vortices = vort[vort_select, :]
            # print('\n', vortices, '\n')

            if len(vortices) > 1:
                # plt.figure()
                # plt.imshow(phi_zoom0, cmap='twilight_shifted', interpolation='none')
                # plt.plot(vortices[:,0] + phi_zoom0.shape[0]//2, vortices[:,1] + phi_zoom0.shape[0]//2, 'x', color='green')
                # plt.show()
                distances = np.sqrt(
                    (vortices[0, 0] - vortices[1, 0]) ** 2
                    + (vortices[0, 1] - vortices[1, 1]) ** 2
                )
                veloo = 1 / (distances * d_real / (xi))
                if veloo > 1:
                    veloo = np.nan
                qs_velo_avg += [veloo]
                marker_s = "s"
            else:
                phi_zoom1 = phi_zoom0.copy()
                # phase_jrs = phi_zoom1[phi_zoom1.shape[0]//2,phi_zoom1.shape[0]//2-pxx:phi_zoom1.shape[0]//2+pxx]
                # phase_jrs = np.mean(phi_zoom1[phi_zoom1.shape[0]//2-2:phi_zoom1.shape[0]//2+2,phi_zoom1.shape[0]//2-pxx:phi_zoom1.shape[0]//2+pxx], axis=0)
                phase_jrs = np.mean(
                    phi_zoom1[
                        phi_zoom1.shape[0] // 2 - 2 : phi_zoom1.shape[0] // 2 + 2, :
                    ],
                    axis=0,
                )
                phase_jrs += np.pi
                x_zoom = np.linspace(-ww, ww, 2 * ww) * size_norm
                popt1, pcov1 = curve_fit(Soliton1D, x_zoom, phase_jrs, p0=[0.6, 1])
                # plt.figure(12122)
                # plt.plot(x_zoom, phase_jrs, marker="o", label="Phase")
                # plt.plot(
                #     x_zoom,
                #     Soliton1D(x_zoom, *popt1) - np.pi / 2,
                #     label=f"v/cs = {popt1[0]:.2f}",
                # )
                # plt.xlabel(r"$x/\xi$")
                # plt.ylabel(r"$\phi$")
                # plt.ylim(0, 2 * np.pi)
                # plt.legend()
                # # plt.grid()
                # plt.show()
                qs_velo_avg += [popt1[0]]
                marker_s = "o"

            # if i==23:
            #     extent = [-w, w, -w, w]
            #     # drhoo = (drho // 0.1) * 0.1
            #     extent = [-w, w, -w, w]
            #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            #     fig.suptitle(f"tau = {taus[i]:.1f}")
            #     im0 = ax[0].imshow(drho, cmap='gray', extent=extent, interpolation='none', vmin=0, vmax=1)
            #     fig.colorbar(im0, ax=ax[0], label=r"$\rho$", shrink=0.6)
            #     im0.set_clim(0, 1)
            #     ax[0].set_xlabel(r'$x/\xi$')
            #     ax[0].set_ylabel(r'$y/\xi$')
            #     ax[0].set_title('Density')
            #     im1 = ax[1].imshow(phi_zoom0, cmap='twilight_shifted', extent=extent, interpolation='none')
            #     fig.colorbar(im1, ax=ax[1], label=r"$\phi$", shrink=0.6)
            #     im1.set_clim(-np.pi, np.pi)
            #     ax[1].set_xlabel(r'$x/\xi$')
            #     ax[1].set_ylabel(r'$y/\xi$')
            #     ax[1].set_title('Phase')
            #     plt.show()
            #     plt.figure()
            #     alpha = np.mean(drho[:, drho.shape[0]//2-2:drho.shape[0]//2+2], axis=1)
            #     beta = np.mean(drho[drho.shape[0]//2-2:drho.shape[0]//2+2, :], axis=0)
            #     x_shape = np.linspace(-w, w, len(alpha))
            #     plt.plot(x_shape, beta, color='tab:red', label='x')
            #     plt.plot(x_shape, alpha, color='tab:blue', label='y')
            #     plt.ylim(0, 1)
            #     plt.show()

        qs_velo_avg = np.array(qs_velo_avg)
        marker += [marker_s]
        qs_velo += [np.nanmean(qs_velo_avg)]
        qs_velo_err += [np.nanstd(qs_velo_avg)]

    qs_velo = np.array(qs_velo)
    qs_velo_err = np.array(qs_velo_err)
    marker = np.array(marker)
    np.save(f"{scan}/qs_velo.npy", qs_velo)

    if plot:
        plt.figure(figsize=(3.5, 5))
        plt.plot(taus, qs_velo, color="tab:blue")
        for i in range(len(marker)):
            plt.errorbar(
                taus[i],
                qs_velo[i],
                qs_velo_err[i],
                marker=marker[i],
                markeredgecolor="tab:blue",
                markerfacecolor="lightsteelblue",
                ecolor="tab:blue",
            )
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$v/c_s$")
        plt.ylim(0, 1)
        plt.axhline(0.61, color="red", linestyle="--")
        plt.grid()
        plt.savefig(f"{scan}/jrs_mach.svg", dpi=300)
        plt.show()

    return qs_velo


def trajectory_fit(
    scan: str,
    start: int = 0,
    windows_size: float = 12.5,
    t0_end: float = 80,
    plot=False,
):
    """Follow the quasi-soliton trajectory and compare to the geometric optics equation."""
    t0 = time.perf_counter()
    # fields_ref = np.load(f"{path_dn}/{scan}/field_ref.npy")
    # fields = np.load(f"{path_dn}/{scan}/field.npy")
    # fields_vortex = np.load(f"{path_dn}/{scan}/field_vortex.npy")
    fields_ref = np.load(f"{scan}/field_ref.npy")
    fields = np.load(f"{scan}/field.npy")
    fields_vortex = np.load(f"{scan}/field_vortex.npy")
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis_final.npy")
    t = time.perf_counter() - t0
    sz = fields_ref.nbytes + fields.nbytes
    rate = sz / t
    print(f"Loaded {sz * 1e-6:.0f} MB of data in {t:.2f} s / {rate * 1e-6:.2f} MB/s")

    size_norms = []
    qs_pos = []
    qs_pos_err = []
    n_ref = []
    n_ref0 = []

    for i in tqdm.tqdm(range(fields.shape[0]), position=1):
        if i < start:
            continue
        qs_pos_avg = []
        n_ref_avg = []
        for j in range(fields.shape[1]):
            # for j in np.array([0,1]):
            field = fields[i, j, :, :].copy()
            field_ref = fields_ref[i, j, :, :].copy()
            field_vortex = fields_vortex[i, j, :, :].copy()
            # Compute the phase and density
            contrast.exp_angle_fast(field, field_ref)
            contrast.exp_angle_fast(field_vortex, field_ref)
            phi_jrs_flat = contrast.angle_fast(field)
            phi_jrs_flat = ndimage.gaussian_filter(phi_jrs_flat, 8)
            rho = field.real * field.real + field.imag * field.imag
            # rho = ndimage.gaussian_filter(rho, 3)
            rho_ref = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            rho_vort = (
                field_vortex.real * field_vortex.real
                + field_vortex.imag * field_vortex.imag
            )
            rho = np.sqrt(rho)
            rho = rho / np.nanmax(rho)
            rho_ref = np.sqrt(rho_ref)
            rho_ref = rho_ref / np.nanmax(rho_ref)
            rho_vort = np.sqrt(rho_vort)
            rho_vort = rho_vort / np.nanmax(rho_vort)

            # Find qs position in xi rescaled window
            w = windows_size
            # xi = np.sqrt(0.2/(taus[i]*8e6))
            xi = xis[i]
            ww = int(w * xi / d_real)
            size_norm = d_real / xi
            rho_zoom0 = rho[
                rho.shape[0] // 2 - ww : rho.shape[0] // 2 + ww,
                rho.shape[1] // 2 - ww : rho.shape[1] // 2 + ww,
            ]
            rho_ref_zoom0 = rho_ref[
                rho.shape[0] // 2 - ww : rho.shape[0] // 2 + ww,
                rho.shape[1] // 2 - ww : rho.shape[1] // 2 + ww,
            ]
            # if i==start:
            #     rho_start = rho_zoom0.copy()
            # if i==18:
            #     rho_mid = rho_zoom0.copy()
            # if i==39:
            #     rho_end = rho_zoom0.copy()
            phi_jrs_flat_zoom = phi_jrs_flat[
                rho.shape[0] // 2 - ww : rho.shape[0] // 2 + ww,
                rho.shape[1] // 2 - ww : rho.shape[1] // 2 + ww,
            ]
            # n_ref += [rho_ref**2]
            v_tot = np.gradient(unwrap_phase(phi_jrs_flat_zoom))
            v_tot_map = np.hypot(v_tot[0], v_tot[1])
            # Find qs position
            inc = np.abs(v_tot_map) ** 2
            inc /= np.nanmax(inc)
            vorticess = np.unravel_index(inc.argmax(), inc.shape)
            center_xx = vorticess[1]
            center_yy = vorticess[0]
            if i == start:
                n_ref0 += [np.abs(rho_ref_zoom0[int(center_yy), int(center_xx)]) ** 2]
            n_ref00 = np.abs(rho_ref_zoom0[int(center_yy), int(center_xx)]) ** 2
            n_ref01 = np.abs(rho_ref_zoom0[int(center_yy), int(center_xx + 1)]) ** 2
            n_ref10 = np.abs(rho_ref_zoom0[int(center_yy + 1), int(center_xx)]) ** 2
            n_ref11 = np.abs(rho_ref_zoom0[int(center_yy + 1), int(center_xx + 1)]) ** 2
            n_ref_avg += [[n_ref00, n_ref01, n_ref10, n_ref11]]
            if (
                (field[field.shape[0] // 2, field.shape[1] // 2] < 0.01)
                or (field_ref[field_ref.shape[0] // 2, field_ref.shape[1] // 2] < 0.01)
                or (
                    field_vortex[field_vortex.shape[0] // 2, field_vortex.shape[1] // 2]
                    < 0.01
                )
            ):
                n_ref_avg += [[np.nan, np.nan, np.nan, np.nan]]
            # Find qs position in not rescaled px window
            v_tot2 = np.gradient(unwrap_phase(phi_jrs_flat))
            v_tot_map2 = np.hypot(v_tot2[0], v_tot2[1])
            # Find qs position
            inc2 = np.abs(v_tot_map2) ** 2
            inc2 /= np.nanmax(inc2)
            vorticess2 = np.unravel_index(inc2.argmax(), inc2.shape)
            center_xx_px = vorticess2[1]
            center_yy_px = vorticess2[0]
            size_norms += [size_norm]
            center_xx = (center_xx - ww) * size_norm
            center_yy = np.abs(center_yy - ww) * size_norm
            if (
                (field[field.shape[0] // 2, field.shape[1] // 2] < 0.01)
                or (field_ref[field_ref.shape[0] // 2, field_ref.shape[1] // 2] < 0.01)
                or (
                    field_vortex[field_vortex.shape[0] // 2, field_vortex.shape[1] // 2]
                    < 0.01
                )
            ):
                center_xx = np.nan
                center_yy = np.nan
            else:
                qs_pos_avg += [[center_xx, center_yy]]
        qs_pos_avg = np.array(qs_pos_avg)
        qs_pos += [np.nanmean(qs_pos_avg, axis=0)]
        qs_pos_err += [np.nanstd(qs_pos_avg, axis=0)]
        n_ref_avg = np.array(n_ref_avg)
        n_ref += [np.nanmean(n_ref_avg, axis=0)]

    size_norms = np.array(size_norms)
    n_ref = np.array(n_ref)
    n_ref0 = np.array(n_ref0)
    n_ref0 = np.nanmean(n_ref0)
    qs_pos = np.array(qs_pos)
    qs_pos_err = np.array(qs_pos_err)

    # plt.figure()
    # plt.plot(qs_pos[:,0], qs_pos[:,1], marker='o')
    # plt.grid()
    # plt.show()

    # rho_start = rho_start[0:rho_start.shape[0]//2, :]
    # rho_mid = rho_mid[0:rho_mid.shape[0]//2, :]
    # rho_end = rho_end[0:rho_end.shape[0]//2, :]

    print("\n", "Fiting trajectory", "\n")

    n_reff_series = n_ref
    # ww = windows_size

    def Qs_smirnov(t, y, e0, n_reff_series):
        index = int(t // (t_end / len(n_reff_series)))
        if index >= len(n_reff_series):
            index = len(n_reff_series) - 1
        print("t_end = ", t, end="\r")
        n_reff = n_reff_series[index]

        # n_reff = np.flip(n_reff, axis=0)
        posx, posy, vx, vy = y
        posx = int(round(posx))
        posy = int(round(posy))

        a_smirnov = 2 * np.pi + ((2 * np.pi) / 3) * np.exp(
            -(((e0 * n_ref0) / (9.8 * n_reff)) ** 2)
        )
        nu_smirnov = (
            np.sqrt(n_reff)
            * (a_smirnov / (e0 * n_ref0))
            * np.sinh((e0 * n_ref0) / (a_smirnov * n_reff))
        )
        nu_smirnov2 = nu_smirnov * nu_smirnov

        drhox = nu_smirnov2[1] - nu_smirnov2[0]
        drhoy = nu_smirnov2[2] - nu_smirnov2[0]
        norm = np.sqrt(vx * vx + vy * vy)
        vx /= norm
        vy /= norm
        vx *= nu_smirnov[0]
        vy *= nu_smirnov[0]
        dvx = 0.5 * drhox
        dvy = -0.5 * drhoy
        dy = [vx, vy, dvx, dvy]
        return dy

    def Qs_fit(t, e0, t_end, n_reff_series):
        t_evals = np.linspace(0, t_end, len(qs_pos[:, 0]))
        res = solve_ivp(
            fun=Qs_smirnov,
            t_span=(0, t_end),
            y0=Z0,
            t_eval=t_evals,
            args=(e0, n_reff_series),
        )
        return res.y[0], res.y[1]

    def fit_function_wrapper(t, e0, t_end, n_reff_series):
        x, y = Qs_fit(t, e0, t_end, n_reff_series)
        return np.concatenate((x, y))

    qs_pos_px = qs_pos
    qs_velo0 = np.array([3.5e-1, 1e-14])
    Z0 = np.array([qs_pos_px[0, 0], qs_pos_px[0, 1], qs_velo0[0], qs_velo0[1]])

    ydata = np.concatenate((qs_pos_px[:, 0], qs_pos_px[:, 1]))
    t_end = t0_end
    t_evals = np.linspace(0, t_end, len(qs_pos[:, 0]))

    initial_guess = [9, t_end]
    bounds = ([0, 0], [np.inf, np.inf])

    def curve_fit_wrapper(t, *params):
        return fit_function_wrapper(t, *params, n_reff_series)

    popt, pcov = optimize.curve_fit(
        curve_fit_wrapper, t_evals, ydata, p0=initial_guess, bounds=bounds
    )

    e0_opt, t_end_opt = popt
    t_end_opt += 1

    t_evals_opt = np.linspace(0, t_end_opt, len(qs_pos[:, 0]))
    sol = solve_ivp(
        Qs_smirnov, (0, t_end_opt), Z0, t_eval=t_evals_opt, args=(e0_opt, n_reff_series)
    )

    print("\n", "Done", "\n")
    # extent = [-w, w, 0, w]
    # fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    # im0 = ax[0].imshow(rho_start, cmap='gray', extent=extent, interpolation='none', vmin=0, vmax=1)
    # # im0 = ax[0].imshow(nu_start, cmap='inferno', extent=extent, interpolation='none')
    # ax[0].plot(qs_pos[:,0], qs_pos[:,1], '--', color='black', label='Position')
    # # ax[0].plot(sol.y[0], sol.y[1], linestyle='-', label=rf'Fit')
    # fig.colorbar(im0, ax=ax[0], label=r"$\sqrt{\rho}$", shrink=0.4)
    # im0.set_clim(0, 1)
    # ax[0].set_xlabel(r'$x/\xi$')
    # ax[0].set_ylabel(r'$y/\xi$')
    # ax[0].set_title(rf"$\tau$ = {taus[0]:.0f}")
    # ax[0].grid()
    # ax[0].legend(loc='upper left')
    # im1 = ax[1].imshow(rho_mid, cmap='gray', extent=extent, interpolation='none', vmin=0, vmax=1)
    # # im1 = ax[1].imshow(nu_mid, cmap='inferno', extent=extent, interpolation='none')
    # ax[1].plot(qs_pos[:,0], qs_pos[:,1], '--', color='black', label='Position')
    # # ax[1].plot(sol.y[0], sol.y[1], linestyle='-', label=rf'Fit')
    # fig.colorbar(im1, ax=ax[1], label=r"$\sqrt{\rho}$", shrink=0.4)
    # im1.set_clim(0, 1)
    # ax[1].set_xlabel(r'$x/\xi$')
    # ax[1].set_ylabel(r'$y/\xi$')
    # ax[1].set_title(rf"$\tau$ = {taus[21]:.0f}")
    # ax[1].grid()
    # ax[1].legend(loc='upper left')
    # im2 = ax[2].imshow(rho_end, cmap='gray', extent=extent, interpolation='none', vmin=0, vmax=1)
    # # im2 = ax[2].imshow(nu_end, cmap='inferno', extent=extent, interpolation='none')
    # ax[2].plot(qs_pos[:,0], qs_pos[:,1], '--', color='black', label='Position')
    # # ax[2].plot(sol.y[0], sol.y[1], linestyle='-', label=rf'Fit')
    # fig.colorbar(im1, ax=ax[2], label=r"$\sqrt{\rho}$", shrink=0.4)
    # im1.set_clim(0, 1)
    # ax[2].set_xlabel(r'$x/\xi$')
    # ax[2].set_ylabel(r'$y/\xi$')
    # ax[2].set_title(rf"$\tau$ = {taus[-2]:.0f}")
    # ax[2].grid()
    # ax[2].legend(loc='upper left')
    # plt.savefig(f"{scan}/trajectory.svg", dpi=300)
    # if plot:
    #   plt.show()

    cm = plt.cm.get_cmap("viridis")
    colors = cm(np.linspace(0, 1, len(taus)))
    np.save(f"{scan}/qs_pos.npy", qs_pos)
    np.save(f"{scan}/qs_pos_err.npy", qs_pos_err)
    np.save(f"{scan}/qs_pos_fit.npy", sol.y)
    plt.figure()
    for i in range(len(qs_pos)):
        plt.errorbar(
            qs_pos[i, 0],
            qs_pos[i, 1],
            xerr=qs_pos_err[i, 0],
            yerr=qs_pos_err[i, 1],
            marker="o",
            alpha=0.5,
            ecolor=colors[i],
            mec=colors[i],
            mfc=colors[i],
        )
    # for i in range(len(taus)):
    # sc = plt.scatter(qs_pos[i,0], qs_pos[i,1], color=colors[i])
    plt.plot(
        sol.y[0], sol.y[1], linestyle="-", color="tab:red", label=r"Geometric optics"
    )
    plt.xlabel(r"$x/\xi$")
    plt.ylabel(r"$y/\xi$")
    plt.legend(loc="upper left")
    norm = plt.Normalize(0, np.max(taus))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.grid()
    plt.savefig(f"{scan}/trajectory_fit.svg", dpi=300)
    if plot:
        plt.show()

    qs_pos_fit = np.array([sol.y[0], sol.y[1]]).T
    qs_pos_fit[:, 0] = (qs_pos_fit[:, 0] - ww) * size_norm
    qs_pos_fit[:, 1] = np.abs(qs_pos_fit[:, 1] - ww) * size_norm

    return qs_pos_fit, e0_opt, t_end_opt


def velocity_map(scan: str, idx_array=np.array([2, 23, 38]), plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data ...")
    t0 = time.perf_counter()
    taus = np.load(f"{scan}/taus.npy")
    delta_n = taus / (k0 * L)
    # xis = 1 / (k0 * np.sqrt(delta_n))
    xis = np.load(f"{scan}/xis_final.npy")
    roi = 15

    field_ref = np.load(f"{scan}/field_ref.npy")
    N_times = field_ref.shape[0]
    N_avg = field_ref.shape[1]
    field = np.load(f"{scan}/field.npy")
    t = time.perf_counter() - t0
    nbr = 0
    which = idx_array
    flow_tot_max = []
    flow_comp_max = []
    flow_inc_max = []
    fields_zoom = []
    fields_ref_zoom = []

    print("\n", "Find maximum value", "\n")
    # for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
    for i in which:
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in range(N_avg):
            field_ref_zoom1 = field_ref[
                i,
                j,
                field_ref.shape[-2] // 2 - roi_xi : field_ref.shape[-2] // 2 + roi_xi,
                field_ref.shape[-1] // 2 - roi_xi : field_ref.shape[-1] // 2 + roi_xi,
            ]
            field_zoom1 = field[
                i,
                j,
                field.shape[-2] // 2 - roi_xi : field.shape[-2] // 2 + roi_xi,
                field.shape[-1] // 2 - roi_xi : field.shape[-1] // 2 + roi_xi,
            ]
            field_ref_zoom1 = field_ref_zoom1[0 : field_ref_zoom1.shape[-2] // 2, :]
            field_zoom1 = field_zoom1[0 : field_zoom1.shape[-2] // 2, :]
            u_tot = np.zeros(
                (field_zoom1.shape[0], field_zoom1.shape[1]),
                dtype=np.float32,
            )
            u_inc = np.zeros(
                (field_zoom1.shape[0], field_zoom1.shape[1]),
                dtype=np.float32,
            )
            u_comp = np.zeros(
                (field_zoom1.shape[0], field_zoom1.shape[1]),
                dtype=np.float32,
            )
            contrast.exp_angle_fast(field_zoom1, field_ref_zoom1)

            filtering_radius = 8
            field_zoom1 = ndimage.gaussian_filter(field_zoom1, filtering_radius)
            fields_zoom += [field_zoom1]
            fields_ref_zoom += [field_ref_zoom1]
            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(field_zoom1, dx=d_real, plot=False)

            flow_tot_max += [np.nanmax(np.hypot(u_tot[0], u_tot[1]))]
            flow_comp_max += [np.nanmax(np.hypot(u_comp[0], u_comp[1]))]
            flow_inc_max += [np.nanmax(np.hypot(u_inc[0], u_inc[1]))]
    flow_tot_max = np.nanmax(np.array(flow_tot_max))
    flow_comp_max = np.nanmax(np.array(flow_comp_max))
    flow_inc_max = np.nanmax(np.array(flow_inc_max))
    fields_zoom = np.array(fields_zoom)
    fields_ref_zoom = np.array(fields_ref_zoom)

    print("\n", "Compute velocity fields", "\n")
    # for i in tqdm.tqdm(range(N_times), desc="Computing energies", position=1):
    for i in which:
        xi = xis[i]
        roi_xi = int(roi * (xi / d_real))
        for j in range(N_avg):
            # field_ref_zoom = field_ref[i,j,field_ref.shape[-2]//2-roi_xi:field_ref.shape[-2]//2+roi_xi, field_ref.shape[-1]//2-roi_xi:field_ref.shape[-1]//2+roi_xi]
            # field_zoom = field[i,j,field.shape[-2]//2-roi_xi:field.shape[-2]//2+roi_xi, field.shape[-1]//2-roi_xi:field.shape[-1]//2+roi_xi]
            field_zoom = fields_zoom[nbr]
            field_ref_zoom = fields_ref_zoom[nbr]
            u_tot = np.zeros(
                (field_zoom.shape[0], field_zoom.shape[1]),
                dtype=np.float32,
            )
            u_inc = np.zeros(
                (field_zoom.shape[0], field_zoom.shape[1]),
                dtype=np.float32,
            )
            u_comp = np.zeros(
                (field_zoom.shape[0], field_zoom.shape[1]),
                dtype=np.float32,
            )
            # contrast.exp_angle_fast(field_zoom, field_ref_zoom)

            # filtering_radius = 8
            # field_zoom = ndimage.gaussian_filter(field_zoom, filtering_radius)

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(field_zoom, dx=d_real, plot=False)
        flow_tot = np.hypot(u_tot[0], u_tot[1]) / flow_tot_max
        flow_comp = np.hypot(u_comp[0], u_comp[1]) / flow_comp_max
        flow_inc = np.hypot(u_inc[0], u_inc[1]) / flow_inc_max
        YY, XX = np.indices(flow_tot.shape)

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=[12, 3])
            fig.suptitle(rf"$\tau$ = {taus[i]:.0f} / $\xi$ = {xi * 1e6:.0f} $\mu m$")
            # flows are calculated by streamplot
            im0 = ax[0].imshow(flow_comp, cmap="viridis", vmin=0, vmax=1)
            ax[0].streamplot(
                XX, YY, u_comp[0], u_comp[1], density=0.5, color="white", linewidth=0.5
            )
            ax[0].set_title(r"$u^{comp}$")
            ax[0].set_xlabel("x")
            ax[0].set_ylabel("y")
            fig.colorbar(im0, ax=ax[0], label=r"$|u^comp}|$")

            im1 = ax[1].imshow(flow_inc, cmap="viridis", vmin=0, vmax=1)
            ax[1].streamplot(
                XX, YY, u_inc[0], u_inc[1], density=0.5, color="white", linewidth=0.5
            )
            ax[1].set_title(r"$u^{inc}$")
            ax[1].set_xlabel("x")
            ax[1].set_ylabel("y")
            fig.colorbar(im1, ax=ax[1], label=r"$|u^inc}|$")
            im2 = ax[2].imshow(flow_tot, cmap="viridis", vmin=0, vmax=1)
            ax[2].streamplot(
                XX, YY, u_tot[0], u_tot[1], density=0.5, color="white", linewidth=0.5
            )
            ax[2].set_title(r"$u^{tot}$")
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("y")
            fig.colorbar(im2, ax=ax[2], label=r"$|u^tot}|$")

            plt.savefig(f"{scan}/velocity_{i}.pdf", dpi=300)
            plt.show()
        nbr += 1


def KPC_compute(scan, plot):
    """Compute JRS shape and compare to KP condition."""
    fields = np.load(f"{scan}/field.npy")
    fields_ref = np.load(f"{scan}/field_ref.npy")
    taus = np.load(f"{scan}/taus.npy")
    xis = np.load(f"{scan}/xis_final.npy")
    qs_velo = np.load(f"{scan}/qs_velo.npy")
    u_inc = np.load(f"{scan}/u_inc.npy")
    JRS_index = np.where(qs_velo > 0.61)
    roi = 15
    x_length = []
    x_length_err = []
    y_length = []
    y_length_err = []
    for i in tqdm.tqdm(np.array(JRS_index).flatten(), desc="Plotting", position=1):
        x_mean_length = []
        y_mean_length = []
        for j in range(fields.shape[1]):
            field = fields[i, j, :, :]
            field = ndimage.gaussian_filter(field, 2)
            field_ref = fields_ref[i, j, :, :]
            field_ref = ndimage.gaussian_filter(field_ref, 2)
            contrast.exp_angle_fast(field, field_ref)
            phi = contrast.angle_fast(field)
            rho = field.real * field.real + field.imag * field.imag
            amp = np.sqrt(rho) ** 2
            amp = amp / np.nanmax(amp)

            # Find qs position
            inc = (np.abs(u_inc[i, :, :, :, :]) ** 2).sum(axis=-3).mean(axis=0)
            inc = ndimage.gaussian_filter(inc, 2)
            inc /= np.nanmax(inc)
            vortices = np.unravel_index(inc.argmax(), inc.shape)
            center_x = vortices[1]
            center_y = vortices[0]
            xi = xis[i]
            size_norm = xi / d_real
            ww = int(roi * xi / d_real)
            amp_zoom = amp[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            phi_zoom = phi[
                int(center_y - ww) : int(center_y + ww),
                int(center_x - ww) : int(center_x + ww),
            ]
            center = (amp_zoom.shape[0] // 2, amp_zoom.shape[0] // 2)
            # figure, ax = plt.subplots(1, 2, figsize=(12, 5))
            # ax[0].imshow(amp_zoom, cmap="gray")
            # ax[1].imshow(phi_zoom, cmap="twilight_shifted")
            # plt.show()
            width = amp_zoom[amp_zoom.shape[0] // 2, :]
            length = amp_zoom[:, amp_zoom.shape[1] // 2]
            mid_height_width = (np.max(width) + np.min(width)) / 2
            mid_height_length = (np.max(length) + np.min(length)) / 2
            index_mid_width = np.abs(width - mid_height_width).argmin()
            index_mid_length = np.abs(length - mid_height_length).argmin()
            # plt.figure()
            # plt.plot(width, label="Width")
            # plt.plot(length, label="Length")
            # plt.axvline(index_mid_width, color="b")
            # plt.axvline(width.argmin(), color="b")
            # plt.axvline(index_mid_length, color="r")
            # plt.axvline(length.argmin(), color="r")
            # plt.legend()
            # plt.show()
            rayon_width = np.abs(index_mid_width - width.argmin()) * d_real / xi
            rayon_length = np.abs(index_mid_length - length.argmin()) * d_real / xi
            x_mean_length += [rayon_width * 2]
            y_mean_length += [rayon_length * 2]
        x_mean_length = np.array(x_mean_length)
        y_mean_length = np.array(y_mean_length)
        x_length += [np.mean(x_mean_length)]
        x_length_err += [np.std(x_mean_length)]
        y_length += [np.mean(y_mean_length)]
        y_length_err += [np.std(y_mean_length)]
    x_length = np.array(x_length)
    y_length = np.array(y_length)
    x_length_err = np.array(x_length_err)
    y_length_err = np.array(y_length_err)
    np.save(f"{scan}/x_length.npy", x_length)
    np.save(f"{scan}/y_length.npy", y_length)
    np.save(f"{scan}/x_length_err.npy", x_length_err)
    np.save(f"{scan}/y_length_err.npy", y_length_err)

    eps = np.sqrt(1 - qs_velo[JRS_index])
    x_kpc = 1 / eps
    y_kpc = 3 / eps**2
    # Calculer l'erreur absolue moyenne en pourcentage (MAPE)
    width_error_pct = np.mean(np.abs((x_length - x_kpc) / x_length)) * 100
    length_error_pct = np.mean(np.abs((y_length - y_kpc) / y_length)) * 100
    print("\n", f"Width error: {width_error_pct:.2f} %")
    print("\n", f"Length error: {length_error_pct:.2f} %")

    plt.figure()
    plt.plot(taus[JRS_index], x_length, marker="o", label="Width")
    plt.fill_between(
        taus[JRS_index],
        x_length - x_length_err,
        x_length + x_length_err,
        alpha=0.4,
    )
    plt.plot(taus[JRS_index], y_length, marker="o", label="Length")
    plt.fill_between(
        taus[JRS_index],
        y_length - y_length_err,
        y_length + y_length_err,
        alpha=0.4,
    )

    plt.plot(taus[JRS_index], x_kpc, "--", color="tab:blue", label=r"$1/\epsilon$")
    plt.plot(taus[JRS_index], y_kpc, "--", color="tab:orange", label=r"$3/\epsilon^2$")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Size in $\xi$ unit")
    plt.legend()
    plt.grid()
    plt.savefig(f"{scan}/size_evolution.svg", dpi=300)
    plt.show()


# %% Manual data processing

if __name__ == "__main__":
    scan_fig1 = "Data/07191910_dipole_time"  # datas of Fig1
    scan_fig2 = "Data/09061956_dipole_time"  # datas of Fig2
    scan_fig3 = "Data/09101636_dipole_time"  # datas of Fig3
    scan_fig4 = "Data/spectral_analysis"  # datas of Fig4

    ### Compute the field from raw images, compute parameters
    # compute_phase_time(scan, plot=True)

    ### Compute system parameters
    # compute_n2_Isat(scan, plot=False)
    # compute_final_xi(scan, plot=True)

    ### plot the fields, energy or velocity and generate an mp4
    # plot_fields(scan, window=15, plot=False)
    # energy_time(scan, plot=False)
    # velocity_map(scan, plot=True)

    ### compute the quasi-soliton velocity, trajectory and velocity correlation
    # normalized_velocity(scan, plot=True)
    # trajectory_fit(scan, start=3, windows_size=12.5, t0_end=130, plot=True)
    # KPC_compute(scan, plot=True)
    # spectral_analysis(scan, which="vortex", img_nbr=4, plot=True, debug=True)
