# %% Import libraries and functions
import contrast
import velocity
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from NLSE import NLSE
from scipy import signal, optimize
from cycler import cycler
from scipy.constants import c, epsilon_0
from skimage import restoration
from matplotlib import colors
from PIL import Image
import scipy.ndimage as ndi
from scipy.integrate import solve_ivp
from PyQt5 import QtWidgets
from matplotlib import animation
from IPython.display import HTML
import matplotlib.ticker as ticker
from tqdm import tqdm
import numba


def power_law(x, a, b):
    """Calcul la loi de puissance avec les constantes a et b"""
    return a * np.power(x, b)


def Flat(XX, YY):
    r = cp.hypot(cp.asarray(simu.YY), cp.asarray(simu.XX))
    theta = cp.arctan2(cp.asarray(simu.YY), cp.asarray(simu.XX))
    Psi = r / cp.sqrt(r**2 + (xi / 0.83) ** 2) * cp.exp(1j * ell * theta)
    Amp = cp.ones((simu.NX, simu.NY))
    Phase = cp.zeros((simu.NX, simu.NY))
    return Amp, Phase


def Vortex(XX, YY, xi=10e-6, ell=1):
    r = cp.hypot(cp.asarray(simu.YY), cp.asarray(simu.XX))
    theta = cp.arctan2(cp.asarray(simu.YY), cp.asarray(simu.XX))
    Psi = r / cp.sqrt(r**2 + (xi / 0.83) ** 2) * cp.exp(1j * ell * theta)
    Amp = cp.abs(Psi)
    Phase = cp.angle(Psi)
    return Amp, Phase


def Soliton(XX, YY, Mach=0.0, xi=10e-6):
    y = cp.asarray(simu.YY)
    x = cp.asarray(simu.XX)
    Psi = (
        cp.sqrt(1 - Mach**2) * cp.tanh((y / (cp.sqrt(2) * xi)) * cp.sqrt(1 - Mach**2))
        + 1j * Mach
    )
    Amp = cp.abs(Psi)
    Phase = cp.angle(Psi)
    return Amp, Phase


def Dipole(XX, YY, dist=100e-6, pos=(0, 0), angle=90, xi=10e-6, ell=1):
    angle = np.pi * angle / 180
    r1 = cp.hypot(cp.asarray(simu.YY) + dist / 2 + pos[1], cp.asarray(simu.XX) + pos[0])
    r2 = cp.hypot(cp.asarray(simu.YY) - dist / 2 + pos[1], cp.asarray(simu.XX) + pos[0])
    theta1 = cp.arctan2(
        cp.asarray(simu.YY) + dist / 2 + pos[1], cp.asarray(simu.XX) + pos[0]
    )
    theta2 = cp.arctan2(
        cp.asarray(simu.YY) - dist / 2 + pos[1], cp.asarray(simu.XX) + pos[0]
    )
    theta1 += angle
    theta2 += angle
    Psi1 = r1 / cp.sqrt(r1**2 + (xi / 0.83) ** 2) * cp.exp(1j * ell * theta1)
    Amp1 = cp.abs(Psi1)
    Phase1 = cp.angle(Psi1)
    Psi2 = r2 / cp.sqrt(r2**2 + (xi / 0.83) ** 2) * cp.exp(1j * -ell * theta2)
    Amp2 = cp.abs(Psi2)
    Phase2 = cp.angle(Psi2)
    Amp = Amp1 + Amp2
    Amp /= cp.max(Amp)
    Phase = Phase1 + Phase2
    return Amp, Phase


@numba.njit(cache=True)
def Qs_propagation(t, y, rho_ref, n200):
    rho_ref = n200 * rho_ref
    posx, posy, vx, vy = y
    posx = int(round(posx))
    posy = int(round(posy))
    rho_ref_s = rho_ref * rho_ref
    drhox = rho_ref_s[posy, posx + 1] - rho_ref_s[posy, posx]
    drhoy = rho_ref_s[posy + 1, posx] - rho_ref_s[posy, posx]
    norm = np.sqrt(vx * vx + vy * vy)
    vx /= norm
    vy /= norm
    vx *= rho_ref[posy, posx]
    vy *= rho_ref[posy, posx]
    dvx = 0.5 * drhox
    dvy = -0.5 * drhoy
    dy = [vx, vy, dvx, dvy]
    return dy


def callback_sample_1(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    if i % save_every == 0:
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z


def callback_sample_2(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    if i % save_every == 0:
        rho_ref = np.abs(A.get())
        phi_ref = np.angle(A.get())
        threshold = 2e-2
        mask = rho_ref < threshold * np.max(rho_ref)
        phi_ref_masked = np.ma.array(phi_ref, mask=mask)
        phi_ref_unwrapped = restoration.unwrap_phase(
            phi_ref_masked, wrap_around=(True, True)
        )
        tau = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z
        tau_samples[i // save_every] = tau


def callback_sample_3(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
    vort_rads: np.ndarray,
) -> None:
    import scipy.optimize as opt

    def Vortex2D(mn, y_val, x_val, xi) -> np.ndarray:
        y, x = mn
        r = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
        theta = np.arctan2(y, x)
        Psi = r / np.sqrt(r**2 + (2 * xi) ** 2)
        Amp = np.abs(Psi)
        return Amp.ravel()

    if i % save_every == 0:
        rho_vort = cp.abs(A)
        phi_flat = cp.angle(A)
        vort = velocity.vortex_detection(phi_flat.get(), plot=False, r=1)
        vort[:, 0] -= phi_flat.shape[-1] // 2
        vort[:, 1] -= phi_flat.shape[-1] // 2
        vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > (300) ** 2)
        vortices = vort[vort_select, :]
        center_x = phi_flat.shape[-1] // 2 + vortices[0][0]
        center_y = phi_flat.shape[-1] // 2 + vortices[0][1]
        vort = cp.array([center_x, center_y])
        window_m = 15
        rho_zoom = rho_vort[
            int(center_y - window_m) : int(center_y + window_m),
            int(center_x - window_m) : int(center_x + window_m),
        ]
        rho_zoom = rho_zoom.get()
        x = np.arange(rho_zoom.shape[1])
        y = np.arange(rho_zoom.shape[0])
        X, Y = np.meshgrid(x, y)
        inital_guess = (window_m, window_m, 5)
        popt, pcov = opt.curve_fit(Vortex2D, (Y, X), rho_zoom.ravel(), p0=inital_guess)
        vortex_reshape = Vortex2D((Y, X), *popt).reshape(window_m * 2, window_m * 2)
        vort_rad = popt[2] * d_real
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z
        vort_rads[i // save_every] = vort_rad


# %% Iputs
n20 = -3.79e-10
Isat0 = 5.05e5
waist = 1.7e-3

N = 2048
window = 15e-3
d_real = 2 * 3.15e-6
puiss = 3
I0 = 2 * puiss / (np.pi * waist**2)
n2 = n20
Isat = Isat0  # saturation intensity in W/m^2
print(f"{n2=:.2e}")
print(f"{Isat*1e-4=:.2f}")
k0 = 2 * np.pi / 780e-9

L_cm = 21
L = L_cm * 1e-2
alpha = 21
# alpha = 0
# nl_lenght = 30e-6
nl_lenght = 0

exponent = 2
ell = 1
mach = 0.0
xi = 20e-6

# dist = 2.5 * xi
dist = 1.7 * xi
dist_px = dist * int(N / window)
pos = np.array([250e-6, 250e-6])
zoom = 300
position = 4

pos_array = np.zeros((N, N))

# %% Initialize simulation
LL = L
print(f"Cell size: {LL=:.2e}m")

simu = NLSE(
    alpha,
    puiss=puiss,
    window=window,
    n2=n2,
    V=None,
    L=LL,
    NX=N,
    NY=N,
    nl_length=nl_lenght,
)
simu.delta_z = 1e-4
simu.n2 = n2
simu.I_sat = Isat

N_samples = L_cm
z_samples = cp.zeros(N_samples + 1)  # +1 to account for the 0th step
qs_samples = cp.zeros((N_samples + 1, 2, 2))
tau_samples = cp.zeros(N_samples + 1)
vort_rads = cp.zeros(N_samples + 1)
E0_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
Evort_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
E_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
N_steps = int(round(LL / simu.delta_z))
save_every = N_steps // N_samples

# %% Compute background field

E0 = (
    cp.exp(
        -(cp.asarray(simu.XX) ** exponent + cp.asarray(simu.YY) ** exponent)
        / waist**exponent
    )
    + 1j * 0
)
# E0 = cp.ones((N, N)) + 1j * 0
# E0[simu.XX**2 + simu.YY**2 > (waist/2)**2] = 0
noise_amp = 0.001
noise_amp = 0
E0 += cp.random.normal(0, noise_amp / 2, E0.shape) + 1j * cp.random.normal(
    0, noise_amp / 2, E0.shape
)

print(f"Simulating E0 with {N_samples} samples...")
E_background = simu.out_field(
    E0,
    LL,
    callback=callback_sample_2,
    callback_args=(E0_samples, z_samples),
    plot=False,
    precision="single",
)

rho_ref_samples = np.abs(E0_samples.get()) ** 2
rho_max = np.nanmax(rho_ref_samples[0])
rho_ref_samples = rho_ref_samples / rho_max * I0
phi_ref_samples = np.angle(E0_samples.get())
print(f"Done")


# %% Compute field

# # amp, phase = Flat(simu.XX, simu.YY)
# amp, phase = Vortex(simu.XX, simu.YY, ell=ell, xi=0.0000000001)
# amp, phase = Soliton(simu.XX, simu.YY, Mach=mach, xi=xi)
amp, phase = Dipole(simu.XX, simu.YY, dist=dist, pos=pos, xi=xi, ell=-1)
# plt.figure()
# plt.imshow(np.abs(amp.get())**2, cmap="gray")
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow(phase.get(), cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
# plt.colorbar()
# plt.show()
E0 *= amp
E0 *= cp.exp(1j * phase)
E0_copy = np.abs(E0.get()) ** 2
E0_copy = E0_copy[
    E0_copy.shape[0] // 2 - zoom : E0_copy.shape[0] // 2 + zoom,
    E0_copy.shape[0] // 2 - zoom : E0_copy.shape[0] // 2 + zoom,
]

print(f"Simulating E with {N_samples} samples...")
E = simu.out_field(
    E0,
    LL,
    callback=callback_sample_1,
    callback_args=(E_samples, z_samples),
    plot=False,
    precision="single",
)
print(f"Done")


# %% Plot fields


def fmt(x, pos) -> str:
    a, b = "{:.0e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


E_samples_0 = E_samples[
    :,
    E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom,
    E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom,
]
E_ref_samples_0 = E0_samples[
    :,
    E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
    E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
]
rho = np.abs(E_samples_0.get()) ** 2
phi = np.angle(E_samples_0.get())
rho_ref = np.abs(E_ref_samples_0.get()) ** 2
phi_ref = np.angle(E_ref_samples_0.get())

# x_pos_zoom = x_pos - rho_ref.shape[-1]//2
# y_pos_zoom = y_pos - rho_ref.shape[-1]//2
# x_pos_zoom = zoom + x_pos_zoom
# y_pos_zoom = zoom + y_pos_zoom

print("Saving images...")
fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout="constrained", dpi=300)
im0 = ax[0].imshow(rho[0], cmap="inferno")
# ax[0].plot(x_pos_zoom, y_pos_zoom, color='grey', linestyle='--')µ
im1 = ax[1].imshow(phi[0], cmap="twilight_shifted", interpolation="none")
# ln, = ax[2].plot(simu.X[simu.NX//2-zoom:simu.NX//2+zoom], rho[0, rho.shape[-2]//2, :])
fig.colorbar(
    im0, ax=ax[0], label="Intensity", format=ticker.FuncFormatter(fmt), shrink=0.6
)
fig.colorbar(im1, ax=ax[1], label="Phase", shrink=0.6)
fig.suptitle("Field evolution through the cell")
ax[0].set_title("Intensity at z=0 mm")
ax[1].set_title("Phase at z=0 mm")
# ax[2].set_title("Density cut at y=0 at z=0 mm")
for i in range(N_samples):
    print(f"Saving image: {i}")
    im0.set_data(rho[i])
    im1.set_data(phi[i])
    # ln.set_data(simu.X[simu.NX//2-zoom:simu.NX//2+zoom], rho[i, rho.shape[-2]//2, :])
    ax[0].set_title(f"Intensity at z={z_samples[i]*1e3:.0f} mm")
    ax[1].set_title(f"Phase at z={z_samples[i]*1e3:.0f} mm")
    # ax[2].set_title(f"Density cut at y=0 at z={z_samples[i]*1e3:.0f} mm")
    im0.set_clim(0, np.max(rho[i]))
    im1.set_clim(-np.pi, np.pi)
    plt.savefig(f"return/output_{i}.jpg", dpi=300)
# plt.show()
print("Done")
