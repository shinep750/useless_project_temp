import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from io import BytesIO
from st_audiorec import st_audiorec

# --- PAGE CONFIG ---
st.set_page_config(page_title="👨‍🍳 Chef Mood Analyzer — Chop-by-Chop", layout="wide")

# --- HEADER ---
st.title("👨‍🍳 Chef Mood Analyzer — Chop-by-Chop")
st.markdown("Detects every chop, predicts the next chop, and compares the sound/timing differences between chops.")

# --- SETTINGS (user-friendly) ---
st.sidebar.header("Analysis Settings")
pre_segment = st.sidebar.slider("Segment start before chop (s)", 0.00, 0.20, 0.05, 0.01)
post_segment = st.sidebar.slider("Segment end after chop (s)", 0.10, 0.60, 0.35, 0.05)
onset_prominence = st.sidebar.slider("Onset prominence (detection sensitivity)", 0.1, 2.0, 0.5, 0.1)
min_chop_distance = st.sidebar.slider("Min chop spacing (s)", 0.10, 0.6, 0.25, 0.05)
liberal_mode = st.sidebar.checkbox("Be liberal when judging consistency (friendly)", value=True)

# --- CONSTANTS ---
N_FFT = 2048
HOP_LENGTH = 512
MAX_FREQ = 1000

# --- AUDIO INPUT ---
input_method = st.radio("Choose input:", ("Upload recording", "Record live"), horizontal=True)
audio_data = None
if input_method == "Upload recording":
    audio_data = st.file_uploader("Upload an audio file (wav, mp3)", type=["wav", "mp3"])
else:
    with st.expander("🔴 Click to record (5-10s recommended)", expanded=True):
        audio_data = st_audiorec()

# --- HELPER FUNCTIONS ---
def load_audio(file_like, method, sr=None):
    """Load audio from file-like or bytes (preserve sr unless None)."""
    if method == "Upload recording":
        y, sr = librosa.load(file_like, sr=sr)
    else:
        y, sr = librosa.load(BytesIO(file_like), sr=sr)
    return y, sr

def detect_onsets(y, sr, hop_length=HOP_LENGTH, n_fft=N_FFT, prominence=0.5, min_dist_s=0.25):
    """Return onset frame indices (peaks) using onset strength."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    onset_env = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
    min_dist_frames = int(min_dist_s * sr / hop_length)
    peaks, _ = find_peaks(onset_env, distance=min_dist_frames, prominence=prominence)
    return peaks, onset_env, S

def extract_chop_features(y, sr, chop_times, pre=0.05, post=0.35):
    """Extract features for each chop segment centered at chop_times (seconds)."""
    features = []
    for t in chop_times:
        start = int(max(0, (t - pre) * sr))
        end = int(min(len(y), (t + post) * sr))
        seg = y[start:end]
        if len(seg) < 10:
            continue
        rms = float(np.mean(librosa.feature.rms(y=seg)))
        amp = float(np.max(np.abs(seg)))
        zc = float(np.mean(librosa.zero_crossings(seg)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        # MFCC vector (summary)
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape (13,)
        features.append({
            "time": float(t),
            "start_sample": start,
            "end_sample": end,
            "amplitude": amp,
            "rms": rms,
            "zero_crossings": zc,
            "centroid": centroid,
            "mfcc_mean": mfcc_mean
        })
    return features

def pairwise_diffs(features):
    """Compute differences between successive chops (amplitude, centroid, mfcc distance)."""
    rows = []
    for i in range(len(features)):
        row = {
            "chop_idx": i+1,
            "time_s": round(features[i]["time"], 3),
            "amplitude": round(features[i]["amplitude"], 5),
            "rms": round(features[i]["rms"], 5),
            "centroid_hz": round(features[i]["centroid"], 2),
            "zero_crossings": round(features[i]["zero_crossings"], 3),
        }
        if i < len(features)-1:
            next_t = features[i+1]["time"]
            interval = next_t - features[i]["time"]
            row["next_time_s"] = round(next_t, 3)
            row["interval_s"] = round(interval, 3)
            # amplitude diff
            amp_diff = features[i+1]["amplitude"] - features[i]["amplitude"]
            row["amp_diff"] = round(amp_diff, 5)
            # centroid diff
            cent_diff = features[i+1]["centroid"] - features[i]["centroid"]
            row["centroid_diff"] = round(cent_diff, 3)
            # MFCC Euclidean distance
            mfcc_dist = float(np.linalg.norm(features[i+1]["mfcc_mean"] - features[i]["mfcc_mean"]))
            row["mfcc_dist"] = round(mfcc_dist, 4)
        else:
            row["next_time_s"] = None
            row["interval_s"] = None
            row["amp_diff"] = None
            row["centroid_diff"] = None
            row["mfcc_dist"] = None
        rows.append(row)
    return pd.DataFrame(rows)

def compute_consistency(intervals, liberal=False):
    """Return a consistency score (0..1) and friendly label.
       Intervals: array of seconds between chops.
       liberal: whether to be kinder in thresholds.
    """
    if len(intervals) == 0:
        return 0.0, "No intervals"
    mean_i = np.mean(intervals)
    std_i = np.std(intervals)
    # primary score: 1 - (std/mean) clipped
    raw = 1 - (std_i / mean_i) if mean_i > 0 else 0.0
    score = float(np.clip(raw, 0.0, 1.0))
    # frequency-based CV for extra friendliness
    freq = 1.0 / intervals
    freq_cv = np.std(freq) / np.mean(freq)
    # Liberal thresholds (more forgiving)
    if liberal:
        thresholds = (0.22, 0.38, 0.55)  # excellent, good, ok
    else:
        thresholds = (0.12, 0.25, 0.40)
    if freq_cv < thresholds[0]:
        label = "Excellent ✓"
    elif freq_cv < thresholds[1]:
        label = "Good ✔"
    elif freq_cv < thresholds[2]:
        label = "Fair"
    else:
        label = "Irregular"
    # If intervals are nearly equal (very low std), bump the score a little in liberal mode
    if liberal and score > 0.7:
        score = min(1.0, score + 0.06)
    return score, label

def determine_mood_from_consistency(score, avg_amp):
    """Optional friendly mood determination - uses consistency and intensity."""
    if score > 0.85 and avg_amp < 0.15:
        return "😌 Zen Master", "Calm, superb rhythm"
    if score > 0.75:
        return "😊 Happy Chef", "Consistent, pleasant rhythm"
    if score > 0.55 and avg_amp > 0.4:
        return "😤 Focused", "Some force but decent rhythm"
    if score <= 0.5 and avg_amp > 0.6:
        return "😡 Aggressive", "Irregular and strong chops"
    return "😟 Stressed", "Irregular rhythm"

# --- PLOTTING ---
def plot_spectrogram_with_chops(S, sr, peaks_frames, hop_length=HOP_LENGTH, max_freq=MAX_FREQ):
    fig, ax = plt.subplots(figsize=(10, 4))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    ax.set_ylim(0, max_freq)
    times = librosa.frames_to_time(peaks_frames, sr=sr, hop_length=hop_length)
    for i, t in enumerate(times):
        ax.axvline(t, color='lime', linestyle='--', alpha=0.85)
        ax.text(t, max_freq*0.05, f"{i+1}", color="white", fontsize=9, ha='center', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round'))
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title("Spectrogram — detected chops labeled")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

# --- MAIN APP FLOW ---
if audio_data is not None:
    st.audio(audio_data)
    with st.spinner("Analyzing chops..."):
        try:
            y, sr = load_audio(audio_data, input_method, sr=None)
            # detect onsets (frames)
            peaks_frames, onset_env, S = detect_onsets(
                y, sr,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
                prominence=onset_prominence,
                min_dist_s=min_chop_distance
            )
            if len(peaks_frames) == 0:
                st.warning("No chops detected — try a louder or clearer recording.")
            else:
                chop_times = librosa.frames_to_time(peaks_frames, sr=sr, hop_length=HOP_LENGTH)
                # extract segment-wise features
                chop_features = extract_chop_features(y, sr, chop_times, pre=pre_segment, post=post_segment)
                if len(chop_features) < 2:
                    st.warning("Need at least 2 chops to analyze timing/differences. Try recording a bit longer.")
                else:
                    # dataframe of per-chop values and pairwise diffs
                    df = pairwise_diffs(chop_features)
                    # compute intervals and metrics
                    intervals = df["interval_s"].dropna().values.astype(float)
                    consistency_score, consistency_label = compute_consistency(intervals, liberal=liberal_mode)
                    avg_amp = float(np.mean([f["amplitude"] for f in chop_features]))
                    mood, mood_reason = determine_mood_from_consistency(consistency_score, avg_amp)

                    # Add predicted next time vs actual judgement: use median interval as prediction
                    median_interval = float(np.median(intervals))
                    predicted_nexts = []
                    on_time_flags = []
                    timing_dev = []
                    for i in range(len(chop_features)):
                        t = chop_features[i]["time"]
                        predicted = t + median_interval
                        if i < len(chop_features)-1:
                            actual_next = chop_features[i+1]["time"]
                            dev = actual_next - predicted
                            timing_dev.append(dev)
                            # relative deviation
                            rel_dev = abs(dev) / median_interval if median_interval > 0 else np.nan
                            # tolerant thresholds: liberal_mode increases tolerance
                            tol = 0.20 if liberal_mode else 0.12
                            on_time = abs(rel_dev) <= tol
                        else:
                            actual_next = None
                            dev = None
                            on_time = None
                        predicted_nexts.append(round(predicted, 3))
                        on_time_flags.append(on_time)
                    # attach to df (shift predicted_nexts so each row's predicted is t + median)
                    df["predicted_next_s"] = predicted_nexts
                    df["on_time_next"] = on_time_flags

                    # UI: top summary
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        # big mood image or emoji (simple)
                        if mood.startswith("😌") or mood.startswith("😊"):
                            img_url = "https://cdn-icons-png.flaticon.com/512/2972/2972054.png"
                        elif mood.startswith("😤") or mood.startswith("😡"):
                            img_url = "https://cdn-icons-png.flaticon.com/512/742/742751.png"
                        else:
                            img_url = "https://cdn-icons-png.flaticon.com/512/1995/1995570.png"
                        st.image(img_url, width=120)
                    with col2:
                        st.markdown(f"## {mood}")
                        st.write(f"**Why?** {mood_reason}")
                        st.markdown(f"**Chops detected:** {len(chop_features)}")
                        st.markdown(f"**Median interval:** {median_interval:.3f} s")
                    with col3:
                        st.markdown("### Consistency")
                        st.progress(min(max(consistency_score, 0.0), 1.0))
                        st.write(f"**{consistency_label}** ({consistency_score:.1%})")
                        if liberal_mode:
                            st.caption("Liberal mode: being friendlier on small timing variations")

                    # Show spectrogram with labeled chops
                    st.pyplot(plot_spectrogram_with_chops(S, sr, peaks_frames, hop_length=HOP_LENGTH, max_freq=MAX_FREQ))

                    # Show the per-chop table (friendly)
                    st.markdown("### Chop-by-chop details")
                    # make more readable table: hide full MFCC vector
                    df_display = df.copy()
                    df_display = df_display[[
                        "chop_idx", "time_s", "next_time_s", "interval_s", "predicted_next_s", "on_time_next",
                        "amplitude", "amp_diff", "rms", "centroid_hz", "centroid_diff", "mfcc_dist"
                    ]]
                    st.dataframe(df_display.style.format({
                        "time_s": "{:.3f}", "next_time_s": "{:.3f}", "interval_s": "{:.3f}",
                        "predicted_next_s": "{:.3f}", "amplitude": "{:.5f}", "amp_diff": "{:.5f}",
                        "rms": "{:.5f}", "centroid_hz": "{:.2f}", "centroid_diff": "{:.3f}", "mfcc_dist": "{:.4f}"
                    }), height=300)

                    # Quick stats
                    st.markdown("### Quick Stats")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Chop Interval", f"{np.mean(intervals):.3f} s", f"±{np.std(intervals):.3f}")
                    c2.metric("Consistency", f"{consistency_score:.1%}", consistency_label)
                    c3.metric("Avg Intensity (amp)", f"{avg_amp:.3f}")

                    # Download CSV of full numeric table (including MFCC summary if user wants)
                    full_rows = []
                    for i, row in df.iterrows():
                        feat = chop_features[int(row["chop_idx"])-1]
                        mm = feat["mfcc_mean"]
                        full_rows.append({
                            **row.to_dict(),
                            **{f"mfcc_{j+1}": float(mm[j]) for j in range(len(mm))}
                        })
                    full_df = pd.DataFrame(full_rows)
                    csv = full_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download full chop data (CSV)", csv, "chop_analysis.csv", "text/csv")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# --- HELP ---
with st.expander("📖 How this works (brief)"):
    st.write("""
    - The app finds *onset peaks* (chop moments) in your recording.
    - For each chop we extract amplitude, RMS, spectral centroid and a short MFCC 'timbre' vector.
    - We compare successive chops (timing, amplitude, centroid, MFCC distance).
    - We compute a consistency score (1 = perfectly steady). Turn on *Liberal mode* to be friendlier to small timing differences.
    """)
