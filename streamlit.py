import streamlit as st
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.signal import savgol_filter

# === Streamlit settings ===
st.set_page_config(layout="wide")
st.title("🔬 Spectroscopy Viewer")

# === Biomarker regions ===
biomarkers = [
    {"label": "Nucleic Acids", "range": (780, 790), "color": "skyblue"},
    {"label": "Proteins", "range": (1000, 1010), "color": "violet"},
    {"label": "Lipids", "range": (1440, 1450), "color": "lightgreen"},
]

# === Simulate cube ===
def generate_fake_cube(height=65, width=65, shifts=500):
    x = np.linspace(400, 1800, shifts)
    cube = np.zeros((height, width, shifts))
    for i in range(height):
        for j in range(width):
            spectrum = (
                np.exp(-0.5 * ((x - 780) / 10)**2) * np.random.uniform(0.8, 1.2) +
                np.exp(-0.5 * ((x - 1005) / 15)**2) * np.random.uniform(0.8, 1.2) +
                np.exp(-0.5 * ((x - 1445) / 12)**2) * np.random.uniform(0.8, 1.2) +
                np.random.normal(0, 0.02, size=shifts)
            )
            cube[i, j, :] = spectrum
    return cube, x

# === File uploader ===
uploaded_file = st.file_uploader("Upload a `.mat` file", type=["mat"])

use_fake_data = False

if uploaded_file is not None:
    try:
        mat_data = sio.loadmat(uploaded_file)
        if 'data' not in mat_data:
            st.error("The uploaded .mat file doesn't contain a 'data' key.")
            use_fake_data = True
        else:
            data_cube = mat_data['data']
            shift_values = np.linspace(400, 1800, data_cube.shape[2])
            st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        use_fake_data = True
else:
    st.info("No file uploaded. Using simulated data.")
    use_fake_data = True

# === Generate fake data if needed ===
if use_fake_data:
    data_cube, shift_values = generate_fake_cube()

height, width, num_shifts = data_cube.shape

# === Select shift index ===
chosen_shift_index = st.slider("Choose Shift Index", 0, num_shifts - 1, 100)
intensity_map = data_cube[:, :, chosen_shift_index]

# === Display Intensity Map ===
st.subheader("🗺 Intensity Map")
fig_map, ax_map = plt.subplots()
im = ax_map.imshow(intensity_map, cmap='hot')
fig_map.colorbar(im, ax=ax_map, label='Intensity')
ax_map.set_title(f'Intensity Map (Shift Index: {chosen_shift_index})')
cursor = Cursor(ax_map, useblit=True, color='white', linewidth=1)
map_placeholder = st.pyplot(fig_map, use_container_width=True)

# === Spectrum plot placeholder ===
spectrum_fig, spectrum_ax = plt.subplots()
spectrum_plot = spectrum_ax.plot([], [], lw=2)[0]
spectrum_ax.set_title("Denoised Spectrum at Selected Pixel")
spectrum_ax.set_xlabel("Shift (cm⁻¹)")
spectrum_ax.set_ylabel("Intensity")
spectrum_plot_placeholder = st.pyplot(spectrum_fig, use_container_width=True)

# === Click Handler ===
def handle_click(x, y):
    raw_spectrum = data_cube[y, x, :]
    smoothed = savgol_filter(raw_spectrum, window_length=21, polyorder=3)

    spectrum_ax.clear()
    spectrum_ax.plot(shift_values, smoothed, lw=2)
    spectrum_ax.set_xlim(shift_values[0], shift_values[-1])
    spectrum_ax.set_ylim(np.min(smoothed), np.max(smoothed) * 1.1)
    spectrum_ax.set_title(f"Smoothed Spectrum at ({x}, {y})")
    spectrum_ax.set_xlabel("Shift (cm⁻¹)")
    spectrum_ax.set_ylabel("Intensity")

    # Add biomarker regions
    for bm in biomarkers:
        spectrum_ax.axvspan(bm["range"][0], bm["range"][1], color=bm["color"], alpha=0.3, label=bm["label"])

    # Avoid duplicate legend entries
    handles, labels = spectrum_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    spectrum_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    spectrum_plot_placeholder.pyplot(spectrum_fig, use_container_width=True)

# === Manual pixel selection ===
st.subheader("🎯 Select a Pixel")
col1, col2 = st.columns(2)
with col1:
    x_pixel = st.number_input("X Pixel", min_value=0, max_value=width - 1, value=32)
with col2:
    y_pixel = st.number_input("Y Pixel", min_value=0, max_value=height - 1, value=32)

if st.button("View Spectrum at Pixel"):
    handle_click(x_pixel, y_pixel)
