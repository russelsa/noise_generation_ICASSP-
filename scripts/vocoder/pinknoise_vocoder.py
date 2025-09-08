import os
import glob
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from textgrid import TextGrid
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def generate_pink_noise(length):
    nrows = 16
    array = np.random.randn(nrows, length)
    array = np.cumsum(array, axis=1)
    pink = np.sum(array, axis=0)
    pink -= np.mean(pink)
    pink /= np.max(np.abs(pink))
    return pink.astype(np.float64)

def extract_pink_sp(fs=16000, fft_size=1024, duration_sec=3.0):
    pink = generate_pink_noise(int(fs * duration_sec))
    f0, timeaxis = pw.harvest(pink, fs, frame_period=10.0)
    sp = pw.cheaptrick(pink, f0, timeaxis, fs, fft_size=fft_size)
    avg_sp = np.mean(sp, axis=0, keepdims=True)
    return avg_sp

avg_sp = extract_pink_sp()

def match_rms(y, target):
    rms_y = np.sqrt(np.mean(y ** 2)) + 1e-8
    rms_target = np.sqrt(np.mean(target ** 2)) + 1e-8
    return y * (rms_target / rms_y)

def fast_amplitude_envelope(x, frame_size=1024):
    return np.sqrt(np.convolve(x**2, np.ones(frame_size)/frame_size, mode='same'))

def pitch_preserved_noise(x, fs=16000):
    x = np.ascontiguousarray(x, dtype=np.float64)
    frame_period = 10.0
    fft_size = 512
    f0, timeaxis = pw.harvest(x, fs, frame_period=frame_period)
    sp = pw.cheaptrick(x, f0, timeaxis, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, timeaxis, fs, fft_size=fft_size)
    avg_sp = np.mean(sp, axis=0)
    sp_flat = np.tile(avg_sp, (len(f0), 1))
    voiced_f0 = f0[f0 > 0]
    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 100.0
    f0_flat = np.where(f0 > 0, mean_f0, 0.0)
    y = pw.synthesize(f0, sp_flat, ap, fs, frame_period)
    y_f0_flat = pw.synthesize(f0_flat, sp_flat, ap, fs, frame_period)
    def match_length(sig, target_len):
        return np.pad(sig, (0, max(0, target_len - len(sig))))[:target_len]
    y = match_length(y, len(x))
    y_f0_flat = match_length(y_f0_flat, len(x))
    return y, y_f0_flat

def process_interval(args):
    wav_channel, interval, fs = args
    start_s, end_s = interval.minTime, interval.maxTime
    label = interval.mark.strip()
    if label == "":
        return None
    start = int(start_s * fs)
    end = int(end_s * fs)
    segment = wav_channel[start:end]
    if len(segment) == 0:
        return None
    envelope = fast_amplitude_envelope(segment)
    y_pitch, y_flat = pitch_preserved_noise(segment, fs)
    min_len = min(len(envelope), len(y_pitch))
    envelope = envelope[:min_len]
    y_pitch = y_pitch[:min_len]
    y_flat = y_flat[:min_len]
    y_pitch_and_intensity = y_pitch * envelope
    y_intensity = y_flat * envelope
    ref = segment[:min_len]
    y_pitch = match_rms(y_pitch, ref)
    y_pitch_and_intensity = match_rms(y_pitch_and_intensity, ref)
    y_intensity = match_rms(y_intensity, ref)
    return start, y_pitch, y_pitch_and_intensity, y_intensity

def process_channel(wav_channel, intervals, fs, channel_name=""):
    total_samples = int(intervals[-1].maxTime * fs)
    pitch_output = np.zeros(total_samples)
    shaped_output = np.zeros(total_samples)
    intensity_output = np.zeros(total_samples)

    args = [(wav_channel, interval, fs) for interval in intervals if interval.mark.strip() != ""]

    for i, interval in enumerate(tqdm(args, desc=f"Segments ({channel_name})", leave=False)):
        result = process_interval(interval)
        if result is None:
            continue
        start, y_pitch, y_shaped, y_intensity = result
        end = start + len(y_pitch)
        pitch_output[start:end] = y_pitch[:end - start]
        shaped_output[start:end] = y_shaped[:end - start]
        intensity_output[start:end] = y_intensity[:end - start]

    return pitch_output, shaped_output, intensity_output

def pitch_preserved_noise_textgrid(input_wav, input_textgrid, output_pitch, output_shaped, output_intensity, fs=16000):
    x, sr = librosa.load(input_wav, sr=fs, mono=False)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError("Expected stereo input")
    left, right = x
    tg = TextGrid.fromFile(input_textgrid)
    tier0, tier1 = tg.tiers[0], tg.tiers[1]

    left_pitch, left_shaped, left_intensity = process_channel(left, tier0.intervals, fs, channel_name="Left")
    right_pitch, right_shaped, right_intensity = process_channel(right, tier1.intervals, fs, channel_name="Right")

    max_len = max(len(left_pitch), len(right_pitch))
    def pad_to(x, length): return np.pad(x, (0, length - len(x))) if len(x) < length else x
    left_pitch = pad_to(left_pitch, max_len)
    right_pitch = pad_to(right_pitch, max_len)
    left_shaped = pad_to(left_shaped, max_len)
    right_shaped = pad_to(right_shaped, max_len)
    left_intensity = pad_to(left_intensity, max_len)
    right_intensity = pad_to(right_intensity, max_len)

    sf.write(output_pitch, np.stack([left_pitch, right_pitch], axis=0).T, fs)
    sf.write(output_shaped, np.stack([left_shaped, right_shaped], axis=0).T, fs)
    sf.write(output_intensity, np.stack([left_intensity, right_intensity], axis=0).T, fs)

def process_one_file(wav_file, input_dir, tg_dir, out_dir_pitch, out_dir_shaped, out_dir_intensity):
    base = os.path.basename(wav_file)
    tg_file = os.path.join(tg_dir, base.replace(".wav", ".TextGrid"))
    out_pitch = os.path.join(out_dir_pitch, base)
    out_shaped = os.path.join(out_dir_shaped, base)
    out_intensity = os.path.join(out_dir_intensity, base)


    print(f"Processing {base}")
    pitch_preserved_noise_textgrid(
        input_wav=wav_file,
        input_textgrid=tg_file,
        output_pitch=out_pitch,
        output_shaped=out_shaped,
        output_intensity=out_intensity,
        fs=16000
    )

if __name__ == "__main__":

    wav_file = "sample_data/clean_audio/1f7e582c-c6bc-46b6-b5a4-e5d78e8a46ac.wav"

    input_dir = "sample_data/clean_audio"
    tg_dir = "sample_data/clean_audio"

    out_dir_pitch =  "sample_data/prosodic_noise/pitch"
    out_dir_shaped = "sample_data/prosodic_noise/pitch_and_intensity"
    out_dir_intensity = "sample_data/prosodic_noise/intensity"

    os.makedirs(out_dir_pitch, exist_ok=True)
    os.makedirs(out_dir_shaped, exist_ok=True)
    os.makedirs(out_dir_intensity, exist_ok=True)

    process_one_file(wav_file, input_dir, tg_dir, out_dir_pitch, out_dir_shaped, out_dir_intensity)
