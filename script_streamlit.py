import streamlit as st
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from io import BytesIO

# === Load .mat file ===
st.title("🧪 Raman Spectrum Viewer")
uploaded_file = st.file_uploader("Upload a .mat file", type=["mat"])

if uploaded_file:
    mat_data = sio.loadmat(uploaded_file)
    raman_cube = mat_data['data']  # shape (height, width, shifts)
    height, width, num_shifts = raman_cube.shape
    shift_values = np.linspace(400, 1800, num_shifts)  # adjust if needed

    # Biomarker shading regions
    biomarkers = [
        {"label": "Nucleic Acids", "range": (780, 790), "color": "skyblue"},
        {"label": "Proteins", "range": (1000, 1010), "color": "violet"},
        {"label": "Lipids", "range": (1440, 1450), "color": "lightgreen"},
    ]

    # === Raman Map Viewer ===
    st.subheader("🗺️ Raman Map")
    chosen_shift_index = st.slider("Choose Shift Index", 0, num_shifts - 1, 100)
    raman_map = raman_cube[:, :, chosen_shift_index]

    fig_map, ax_map = plt.subplots()
    im = ax_map.imshow(raman_map, cmap='hot')
    ax_map.set_title(f'Raman Map (Shift Index: {chosen_shift_index})')
    fig_map.colorbar(im, ax=ax_map, label="Intensity")

    st.pyplot(fig_map)

    # === Choose pixel ===
    st.subheader("🔍 Select Pixel to View Spectrum")
    x = st.number_input("X coordinate", min_value=0, max_value=width-1, value=32)
    y = st.number_input("Y coordinate", min_value=0, max_value=height-1, value=32)

    if st.button("Show Spectrum"):
        raw_spectrum = raman_cube[int(y), int(x), :]
        smoothed = savgol_filter(raw_spectrum, window_length=21, polyorder=3)

        fig_spec, ax_spec = plt.subplots()
        ax_spec.plot(shift_values, smoothed, lw=2, label="Denoised Spectrum")
        ax_spec.set_title(f"Spectrum at Pixel ({x}, {y})")
        ax_spec.set_xlabel("Raman Shift (cm⁻¹)")
        ax_spec.set_ylabel("Intensity")

        # Add shaded biomarker regions
        for bm in biomarkers:
            ax_spec.axvspan(bm["range"][0], bm["range"][1], color=bm["color"], alpha=0.3, label=bm["label"])

        # Unique legend
        handles, labels = ax_spec.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_spec.legend(by_label.values(), by_label.keys(), loc='upper right')

        st.pyplot(fig_spec)
