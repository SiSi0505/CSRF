import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Colors (same style as your friend's script) ---
Blue   = (0, 101/255, 189/255)
Green  = (52/255, 129/255,  65/255)
Orange = (241/255,  89/255,  41/255)
Red    = (196/255,  30/255,  58/255)

# --- File path (your uploaded file) ---
VCVS_FILE = "/mnt/data/lna_singleend.vcsv"

# --- Read Cadence .vcsv ---
# Cadence exports header lines starting with ';' and then numeric CSV data.
df = pd.read_csv(VCVS_FILE, comment=';', header=None)

# Data layout per row (8 columns):
# [freq, S11_dB, freq, S12_dB, freq, S21_dB, freq, S22_dB]
freq = df.iloc[:, 0].to_numpy()
s11  = df.iloc[:, 1].to_numpy()
s12  = df.iloc[:, 3].to_numpy()
s21  = df.iloc[:, 5].to_numpy()
s22  = df.iloc[:, 7].to_numpy()

# --- Plot settings ---
f0 = 915e6  # operating frequency
out_file = "lna_sparameters.pdf"  # change name if you want (.png also works)

# --- Plot ---
plt.figure(figsize=(9, 5.5))

plt.semilogx(freq, s11, label=r"$S_{11}$", color=Red)
plt.semilogx(freq, s21, label=r"$S_{21}$", color=Blue)
plt.semilogx(freq, s12, label=r"$S_{12}$", color=Orange)
plt.semilogx(freq, s22, label=r"$S_{22}$", color=Green)

# Mark f0
plt.axvline(f0, linestyle="--", color=(0.35, 0.35, 0.35), linewidth=1.2,
            label=r"$f_0=915\,\mathrm{MHz}$")

# Annotate S21 at f0
idx0 = int(np.argmin(np.abs(freq - f0)))
plt.scatter([freq[idx0]], [s21[idx0]], s=35, color=Blue, zorder=5)
plt.text(freq[idx0]*1.05, s21[idx0], f"S21@f0 = {s21[idx0]:.2f} dB",
         fontsize=10, va="center")

plt.title("LNA S-Parameters (Cadence SpectreRF Export)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()

# Optional axis limits (adjust if you want)
plt.xlim(1e6, 2e9)

plt.tight_layout()
plt.savefig(out_file, dpi=300)  # dpi ignored for pdf but fine
plt.show()

print(f"Saved: {out_file}")
