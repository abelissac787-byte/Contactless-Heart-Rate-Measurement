"""
Contactless heart-rate estimation from a webcam using remote
photoplethysmography (rPPG) with Eulerian Video Magnification (EVM).

Course project for PBECT504 (DSP).

Pipeline (see README for the full rationale):
    capture -> Haar-cascade face detection -> forehead ROI
    -> (optional) EVM colour magnification on the ROI sequence
    -> mean green-channel signal -> detrend/normalise
    -> 4th-order Butterworth band-pass (filtfilt)
    -> sliding-window FFT -> dominant peak -> BPM

Design notes on independence from the public cvzone reference:
    - Face detection uses plain OpenCV Haar cascades (cv2.CascadeClassifier),
      not cvzone/MediaPipe.
    - The frequency axis is built from the *measured* runtime frame rate,
      never a hardcoded assumption.
    - ROI is restricted to the forehead band, band is 0.75-3.5 Hz, and an
      explicit scipy detrend precedes a zero-phase Butterworth stage. These
      are deliberate choices described in the brief, not copied parameters.
    - The EVM idea itself is adapted from Wu et al. (2012); this is cited in
      the README. No code from the reference script was reused.

Usage:
    python heart_rate.py                 # live webcam, EVM on
    python heart_rate.py --no-evm        # live webcam, lightweight (no EVM)
    python heart_rate.py --video clip.mp4
    python heart_rate.py --save-video magnified.mp4
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend

# ---------------------------------------------------------------------------
# Configuration constants (own choices; see brief "Explicit constraints")
# ---------------------------------------------------------------------------
BAND_LOW_HZ = 0.75          # 45 BPM
BAND_HIGH_HZ = 3.5          # 210 BPM
ROI_SIZE = 96               # ROI is resampled to ROI_SIZE x ROI_SIZE px
PYRAMID_LEVELS = 3          # Gaussian pyramid depth used inside EVM
EVM_AMPLIFY = 40.0          # colour amplification factor (tune 20-50)
WINDOW_SECONDS = 12.0       # sliding analysis window length
UPDATE_SECONDS = 1.0        # how often the BPM estimate is recomputed
MIN_SECONDS_TO_ESTIMATE = 6.0   # need at least this much history first
BUTTER_ORDER = 4


# ---------------------------------------------------------------------------
# Signal-processing helpers
# ---------------------------------------------------------------------------
def butter_bandpass(low_hz, high_hz, fs, order=BUTTER_ORDER):
    """Design a Butterworth band-pass; guard the Nyquist limit."""
    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    low = max(low, 1e-4)
    return butter(order, [low, high], btype="band")


def bandpass_trace(trace, fs):
    """Detrend, normalise and zero-phase band-pass a 1-D signal."""
    trace = detrend(np.asarray(trace, dtype=np.float64))
    std = trace.std()
    if std > 1e-8:
        trace = trace / std
    # filtfilt needs a bit more than 3*max(len(a),len(b)) samples to run.
    if len(trace) < 3 * (BUTTER_ORDER + 1):
        return trace
    b, a = butter_bandpass(BAND_LOW_HZ, BAND_HIGH_HZ, fs)
    return filtfilt(b, a, trace)


def dominant_bpm(trace, fs):
    """Return (bpm, freqs_hz, spectrum, peak_index) from an FFT of trace.

    The frequency axis is derived from the *measured* fs, so an incorrect
    assumed frame rate cannot silently corrupt the result.
    """
    n = len(trace)
    if n < 8:
        return None, None, None, None
    windowed = trace * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not band.any():
        return None, freqs, spectrum, None
    band_spectrum = np.where(band, spectrum, 0.0)
    peak = int(np.argmax(band_spectrum))
    bpm = freqs[peak] * 60.0
    return bpm, freqs, spectrum, peak


# ---------------------------------------------------------------------------
# Eulerian Video Magnification (colour) on a stack of ROI frames
# ---------------------------------------------------------------------------
def gaussian_downsample(frame, levels):
    """Repeatedly pyrDown a frame `levels` times; return the smallest level."""
    small = frame
    for _ in range(levels):
        small = cv2.pyrDown(small)
    return small


def gaussian_upsample(frame, levels, target_shape):
    """pyrUp `levels` times, then fit exactly to target (h, w)."""
    big = frame
    for _ in range(levels):
        big = cv2.pyrUp(big)
    h, w = target_shape[:2]
    if big.shape[0] != h or big.shape[1] != w:
        big = cv2.resize(big, (w, h))
    return big


def magnify_roi_stack(roi_stack, fs, levels=PYRAMID_LEVELS,
                      amplify=EVM_AMPLIFY):
    """Apply EVM colour magnification to a temporal stack of ROI frames.

    roi_stack : (T, H, W, 3) float array, values ~[0, 255]
    Returns a magnified stack of the same shape.

    Temporal filtering here is a zero-phase Butterworth applied along the
    time axis of the *downsampled* pyramid level, matching the brief's
    "band-pass each pixel's time series" step. Working on the small level
    keeps this affordable on a CPU.
    """
    t = roi_stack.shape[0]
    if t < 3 * (BUTTER_ORDER + 1):
        return roi_stack  # not enough temporal samples to filter safely

    # Build the downsampled temporal cube.
    small0 = gaussian_downsample(roi_stack[0].astype(np.float32), levels)
    sh, sw = small0.shape[:2]
    cube = np.empty((t, sh, sw, 3), dtype=np.float32)
    cube[0] = small0
    for i in range(1, t):
        cube[i] = gaussian_downsample(roi_stack[i].astype(np.float32), levels)

    # Temporal band-pass along axis 0 (per pixel, per channel).
    b, a = butter_bandpass(BAND_LOW_HZ, BAND_HIGH_HZ, fs)
    filtered = filtfilt(b, a, cube, axis=0)
    filtered *= amplify

    # Reconstruct: original ROI + upsampled amplified residual.
    out = np.empty_like(roi_stack)
    for i in range(t):
        residual = gaussian_upsample(filtered[i], levels, roi_stack[i].shape)
        out[i] = roi_stack[i] + residual
    return out


# ---------------------------------------------------------------------------
# Forehead ROI from a Haar face box
# ---------------------------------------------------------------------------
def forehead_roi(frame, face_box):
    """Crop the forehead / upper-cheek band (upper third, central 60%)."""
    x, y, w, h = face_box
    # Vertical: upper third of the face, nudged down slightly off the hairline.
    y0 = int(y + 0.12 * h)
    y1 = int(y + 0.34 * h)
    # Horizontal: central 60% to avoid temples/hair edges.
    x0 = int(x + 0.20 * w)
    x1 = int(x + 0.80 * w)
    y0, y1 = max(y0, 0), min(y1, frame.shape[0])
    x0, x1 = max(x0, 0), min(x1, frame.shape[1])
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None, None
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Live plotting (matplotlib, non-blocking)
# ---------------------------------------------------------------------------
class LivePlots:
    """Four-panel live view: raw ROI signal, filtered signal, spectrum, BPM."""

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(4, 1, figsize=(7, 8))
        self.fig.suptitle("rPPG heart-rate estimation")
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.bpm_hist = deque(maxlen=120)
        self.bpm_time = deque(maxlen=120)

    def update(self, raw, filtered, freqs, spectrum, peak, bpm, t_now, evm_on):
        a = self.ax
        for axi in a:
            axi.clear()

        a[0].plot(raw, color="tab:gray", lw=0.8)
        a[0].set_title("Raw ROI green signal ({} EVM)".format(
            "with" if evm_on else "no"))

        a[1].plot(filtered, color="tab:blue", lw=0.9)
        a[1].set_title("Detrended + Butterworth band-passed signal")

        if freqs is not None:
            band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
            a[2].plot(freqs[band] * 60.0, spectrum[band], color="tab:green")
            if peak is not None:
                a[2].axvline(freqs[peak] * 60.0, color="tab:red", ls="--")
            a[2].set_title("Spectrum in band (x-axis = BPM)")
            a[2].set_xlabel("BPM")

        if bpm is not None:
            self.bpm_hist.append(bpm)
            self.bpm_time.append(t_now)
        if self.bpm_hist:
            a[3].plot(list(self.bpm_time), list(self.bpm_hist),
                      "o-", color="tab:red", ms=3)
        a[3].set_title("Estimated BPM over time")
        a[3].set_xlabel("seconds")
        a[3].set_ylabel("BPM")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=120)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", help="path to a recorded video file")
    src.add_argument("--camera", type=int, default=0,
                     help="webcam index (default 0)")
    evm = parser.add_mutually_exclusive_group()
    evm.add_argument("--evm", dest="evm", action="store_true",
                     help="enable EVM magnification (default)")
    evm.add_argument("--no-evm", dest="evm", action="store_false",
                     help="disable EVM (lightweight mode)")
    parser.set_defaults(evm=True)
    parser.add_argument("--save-video", default=None,
                        help="write the magnified ROI to this .mp4")
    parser.add_argument("--save-plot", default="rppg_summary.png",
                        help="PNG saved on exit with the final plots")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the live matplotlib window")
    args = parser.parse_args()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit("Could not load Haar cascade at %s" % cascade_path)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    plots = None if args.no_plot else LivePlots()

    # Rolling buffers (bounded once we know the frame rate).
    roi_buffer = deque()          # ROI frames resized to ROI_SIZE
    signal_buffer = deque()       # raw green means (pre-EVM)
    stamp_buffer = deque()        # capture timestamps

    fps_est = 15.0                # provisional; replaced by measurement
    writer = None
    last_update = 0.0
    start = time.time()
    last_box = None               # reuse last face box if detection blinks

    print("Running. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream / camera read failed.")
                break
            now = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                # Largest face wins.
                last_box = max(faces, key=lambda b: b[2] * b[3])

            display = frame.copy()
            if last_box is not None:
                roi, coords = forehead_roi(frame, last_box)
                if roi is not None:
                    x0, y0, x1, y1 = coords
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 0), 2)
                    roi_small = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
                    roi_buffer.append(roi_small.astype(np.float32))
                    signal_buffer.append(float(roi_small[..., 1].mean()))
                    stamp_buffer.append(now)

            # Measure achieved frame rate from timestamps.
            if len(stamp_buffer) >= 2:
                span = stamp_buffer[-1] - stamp_buffer[0]
                if span > 0:
                    fps_est = (len(stamp_buffer) - 1) / span

            # Trim buffers to the analysis window.
            max_frames = max(int(WINDOW_SECONDS * fps_est), 30)
            while len(roi_buffer) > max_frames:
                roi_buffer.popleft()
                signal_buffer.popleft()
                stamp_buffer.popleft()

            elapsed = now - start
            have_seconds = (stamp_buffer[-1] - stamp_buffer[0]
                            if len(stamp_buffer) >= 2 else 0.0)

            bpm_text = "Collecting data..."
            if (have_seconds >= MIN_SECONDS_TO_ESTIMATE
                    and now - last_update >= UPDATE_SECONDS
                    and len(roi_buffer) >= 30):
                last_update = now

                if args.evm:
                    stack = np.stack(roi_buffer, axis=0)
                    magnified = magnify_roi_stack(stack, fps_est)
                    green = magnified[..., 1].reshape(magnified.shape[0], -1)
                    extracted = green.mean(axis=1)
                    latest_roi = np.clip(magnified[-1], 0, 255).astype(np.uint8)
                else:
                    extracted = np.array(signal_buffer, dtype=np.float64)
                    latest_roi = np.clip(roi_buffer[-1], 0, 255).astype(np.uint8)

                filtered = bandpass_trace(extracted, fps_est)
                bpm, freqs, spectrum, peak = dominant_bpm(filtered, fps_est)

                if bpm is not None:
                    bpm_text = "BPM: %.1f  (fps %.1f, %s)" % (
                        bpm, fps_est, "EVM" if args.evm else "raw")

                if plots is not None:
                    plots.update(extracted, filtered, freqs, spectrum,
                                 peak, bpm, elapsed, args.evm)

                # Optional magnified-ROI video artifact.
                if args.save_video:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            args.save_video, fourcc, max(fps_est, 5.0),
                            (ROI_SIZE, ROI_SIZE))
                    writer.write(latest_roi)

            cv2.putText(display, bpm_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(display, bpm_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
            cv2.imshow("rPPG heart rate", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if plots is not None:
            try:
                plots.save(args.save_plot)
                print("Saved summary plot to %s" % args.save_plot)
            except Exception as exc:  # pragma: no cover - best effort
                print("Could not save plot: %s" % exc)


if __name__ == "__main__":
    main()
