# Contactless Heart-Rate Estimation (Webcam rPPG)

DSP course project (PBECT504). Estimates heart rate in BPM from a webcam feed
using remote photoplethysmography (rPPG) with Eulerian Video Magnification.

## Method

```
webcam frame
    -> Haar-cascade face detection (OpenCV)
    -> forehead ROI (upper third, central 60%)
    -> [optional] EVM colour magnification on the ROI sequence
    -> mean green-channel intensity per frame  (1-D signal)
    -> scipy.signal.detrend + normalise
    -> 4th-order Butterworth band-pass, 0.75-3.5 Hz, filtfilt (zero-phase)
    -> sliding-window FFT (~12 s), dominant peak in band
    -> BPM = peak_freq_hz x 60
```

The frequency axis is built from the **measured** runtime frame rate
(timestamp-based, rolling), never a hardcoded value — this is the single most
important correctness point for rPPG and the main flaw in the public reference
implementation.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, `scipy`, `matplotlib`.

## Run

```bash
python heart_rate.py                    # live webcam, EVM on
python heart_rate.py --no-evm           # lightweight mode, EVM off
python heart_rate.py --video clip.mp4   # process a recorded file
python heart_rate.py --save-video magnified.mp4   # save magnified ROI clip
python heart_rate.py --no-plot          # skip the matplotlib window
```

Press `q` in the video window to quit. A four-panel summary plot (raw signal,
band-passed signal, in-band spectrum with the peak marked, and BPM over time)
is shown live and saved to `rppg_summary.png` on exit.

Sit still in even lighting and wait ~6–12 s for the first estimate; it then
updates about once per second over a sliding window.

## Validate the DSP without a camera

```bash
python test_signal_chain.py
```

Feeds synthetic noisy pulse traces (with lighting drift) at several frame
rates and confirms the estimator recovers the known BPM within ±5. All cases
currently pass exactly.

### Manual sanity check (optional, per brief)

Count your own pulse for 30 s, double it, and compare against the on-screen
BPM. In reasonable lighting they should agree within roughly ±5–10 BPM.

## Design choices (and their DSP rationale)

- **Forehead ROI**, not whole face — excludes eye/mouth/hair motion noise.
- **Band 0.75–3.5 Hz (45–210 BPM)** — wide enough to cover elevated rates
  instead of failing silently outside a narrow 60–120 window.
- **Explicit detrending before the band-pass** — removes slow lighting drift
  that a band-pass mask alone leaves as spectral leakage.
- **`filtfilt`** — zero phase, so the trace isn't time-shifted.
- **EVM on the cropped, downsampled ROI only**, ≤3 pyramid levels — keeps the
  magnification affordable on a laptop CPU.

## Known limitations

rPPG is sensitive to lighting, motion, and skin tone. The estimator reports the
strongest in-band frequency, so under poor conditions it can return a
confident-looking but meaningless number. Treat it as a demonstration, not a
medical measurement.

## References

- **H.-Y. Wu, M. Rubinstein, E. Shih, J. Guttag, F. Durand, W. Freeman,
  "Eulerian Video Magnification for Revealing Subtle Changes in the World,"
  ACM Transactions on Graphics (SIGGRAPH), 2012.** — source of the EVM concept.
  Project page: http://people.csail.mit.edu/mrub/evm/
