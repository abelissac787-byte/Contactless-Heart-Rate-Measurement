"""Offline sanity check of the DSP chain, independent of any webcam.

Synthesises a noisy green-channel trace containing a known pulse frequency
and confirms the estimator recovers the correct BPM. Run:

    python test_signal_chain.py
"""

import numpy as np

from heart_rate import bandpass_trace, dominant_bpm


def synthetic_trace(bpm, fs, seconds, noise=0.5, drift=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    hr_hz = bpm / 60.0
    pulse = np.sin(2 * np.pi * hr_hz * t)
    slow_drift = drift * np.sin(2 * np.pi * 0.05 * t)   # lighting drift
    return pulse + slow_drift + noise * rng.standard_normal(t.size)


def run_case(true_bpm, fs, seconds):
    raw = synthetic_trace(true_bpm, fs, seconds)
    filtered = bandpass_trace(raw, fs)
    est, _, _, _ = dominant_bpm(filtered, fs)
    err = abs(est - true_bpm)
    status = "PASS" if err <= 5.0 else "FAIL"
    print(f"[{status}] fs={fs:5.1f}  true={true_bpm:3d}  "
          f"est={est:6.1f}  err={err:4.1f}")
    return err <= 5.0


if __name__ == "__main__":
    cases = [
        (60, 15.0, 15),
        (72, 15.0, 15),
        (90, 30.0, 12),
        (120, 30.0, 12),
        (48, 20.0, 15),
        (180, 30.0, 12),
    ]
    ok = all(run_case(b, fs, s) for b, fs, s in cases)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)
