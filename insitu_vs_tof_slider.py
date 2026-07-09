import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(layout="wide")


st.title("BEC wavefunction: density modulation and momentum peaks")


st.markdown(
"""
### Physical picture

We consider a one-dimensional model of a Bose-Einstein condensate with a
coherent contribution from the zero-momentum state and a pair of symmetric
recoiling momentum states.

The wavefunction is:
"""
)

st.latex(
r"""
\psi(x)=e^{-x^2/(2\sigma^2)}
\left(c_0+c_1\cos(k_0x)\right)
"""
)


st.markdown(
"""
where:

- The Gaussian term describes the condensate envelope.
- \(c_0\) is the population in the original zero-momentum mode.
- \(c_1\) is the population transferred to the recoiling modes.
- \(k_0\) is the recoil momentum.

The cosine term represents a coherent superposition of two momentum states:
"""
)


st.latex(
r"""
\cos(k_0x)=\frac{1}{2}
\left(e^{ik_0x}+e^{-ik_0x}\right)
"""
)


st.markdown(
"""
Therefore, the Fourier transform should show three momentum components:

- a central peak at \(k=0\),
- two symmetric recoil peaks at \(k=\pm k_0\).

The simulation illustrates the connection between:

1. **In situ density modulation**  
   caused by interference between the condensate and recoiling components,

2. **Momentum distribution**  
   obtained from the Fourier transform of the wavefunction.

Use the slider to change the transferred population and observe how the
density contrast and momentum peaks evolve.
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
psi = envelope * (c1 * np.abs(np.cos(k0 * x)))
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
