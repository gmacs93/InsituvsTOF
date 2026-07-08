import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(layout="wide")


st.title("BEC wavefunction: In-situ density modulation vs momentum peaks")

st.markdown(
r"""
### Physical picture

We consider a simple one-dimensional model of a Bose-Einstein condensate
with a coherent contribution from the zero-momentum state and a pair of
symmetric recoiling momentum states:

\[
\psi(x)=e^{-x^2/(2\sigma^2)}
\left(c_0+c_1\cos(k_0x)\right)
\]

where:

- \(e^{-x^2/(2\sigma^2)}\) is the Gaussian envelope of the condensate,
- \(c_0=\sqrt{1-p}\) is the population in the original condensate mode,
- \(c_1=\sqrt{p}\) is the population transferred to the recoiling modes,
- \(k_0\) is the recoil momentum.

The cosine term corresponds to a coherent superposition of two momentum
states:

\[
\cos(k_0x)=\frac{1}{2}(e^{ik_0x}+e^{-ik_0x})
\]

so the momentum distribution should contain peaks around:

\[
k=0,\quad k=\pm k_0 .
\]

The purpose of this simulation is to visualize the relation between:

1. **In situ density modulation**  
   The interference between the zero-momentum and recoiling components creates
   spatial density fringes.

2. **Momentum-space distribution**  
   The Fourier transform reveals the underlying momentum components.

The slider changes the fraction of atoms transferred into the recoiling
state and shows how the density contrast and momentum peaks evolve.
"""
)

# -------------------------
# Parameters
# -------------------------
N = 2048
L = 60

sigma = 8
k0 = 2.0

p = st.slider(
    "Recoiling population",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
)


# -------------------------
# Grid
# -------------------------
x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]

envelope = np.exp(-(x**2) / (2 * sigma**2))

# momentum axis
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))


# -------------------------
# Wavefunction
# -------------------------
c0 = np.sqrt(1 - p)
c1 = np.sqrt(p)

# coherent superposition of 0 and ±k0 momentum states
psi = envelope * (c0 + c1 * np.cos(k0 * x))

density = np.abs(psi)**2


# -------------------------
# Momentum distribution
# -------------------------
psi_k = np.fft.fftshift(np.fft.fft(psi))
momentum = np.abs(psi_k)**2
momentum /= momentum.max()


# -------------------------
# Contrast
# -------------------------
contrast = (
    density.max() - density.min()
) / (
    density.max() + density.min()
)


# -------------------------
# Plot
# -------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 4)
)


ax1.plot(x, density)
ax1.set_xlabel("x")
ax1.set_ylabel(r"$|\psi(x)|^2$")
ax1.set_title(
    f"In situ density\ncontrast = {contrast:.2f}"
)
ax1.grid()


ax2.plot(k, momentum)
ax2.set_xlim(-5, 5)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel(r"$k$")
ax2.set_ylabel(r"$|\psi(k)|^2$")
ax2.set_title("Momentum distribution")
ax2.grid()


st.pyplot(fig)
