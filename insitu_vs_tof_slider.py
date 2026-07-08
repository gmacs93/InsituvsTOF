import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Grid
N = 2048
L = 60

x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]

sigma = 8
envelope = np.exp(-(x**2) / (2 * sigma**2))

k0 = 2.0

# momentum axis
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.18)

slider_ax = plt.axes([0.15, 0.07, 0.7, 0.04])

slider = Slider(
    slider_ax,
    "Recoiling population",
    0,
    1,
    valinit=0.1,
)


def update(p):
    c0 = np.sqrt(1 - p)
    c1 = np.sqrt(p)

    # symmetric momentum state
    psi = envelope * (c0 + c1 * np.cos(k0 * x))
    density = np.abs(psi) ** 2

    # Fourier transform
    psi_k = np.fft.fftshift(np.fft.fft(psi))
    momentum = np.abs(psi_k) ** 2
    momentum /= momentum.max()

    contrast = (density.max() - density.min()) / (density.max() + density.min())

    ax1.clear()
    ax2.clear()

    ax1.plot(x, density)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\psi(x)|^2$")
    ax1.set_title(f"In situ density\nContrast={contrast:.2f}")
    ax1.grid()

    ax2.plot(k, momentum)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$|\psi(k)|^2$")
    ax2.set_title("Momentum distribution")
    ax2.grid()

    fig.canvas.draw_idle()


update(slider.val)
slider.on_changed(update)

plt.show()
