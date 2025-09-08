import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import pyworld as pw
from tqdm import tqdm
import os

def generate_pink_noise(length):
    n_rows = 16
    n = int(length)
    array = np.random.randn(n_rows, n)
    array = np.cumsum(array, axis=1)
    array = array - np.mean(array, axis=1, keepdims=True)
    pink = np.sum(array, axis=0)
    return pink / np.max(np.abs(pink))

def pitch_modulated_noise(x, fs):
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, timeaxis = pw.harvest(x, fs, frame_period=5.0)
    pink = generate_pink_noise(len(x))
    output = np.zeros_like(x)

    hop_size = int(fs * 0.005)
    win_size = int(fs * 0.03)
    win_size += win_size % 2  # ensure odd

    for i, f in enumerate(f0):
        if f < 50:
            continue
        center = int(i * hop_size)
        start = max(center - win_size // 2, 0)
        end = min(center + win_size // 2, len(pink))
        segment = pink[start:end]
        if len(segment) < 3:
            continue

        bw = f * 0.1
        low = max(f - bw / 2, 10) / (fs / 2)
        high = min(f + bw / 2, fs / 2 - 10) / (fs / 2)
        if low >= high:
            continue
        b, a = scipy.signal.butter(2, [low, high], btype='band')
        filtered = scipy.signal.lfilter(b, a, segment)

        output[start:end] += filtered[:end - start]

    output = output / (np.max(np.abs(output)) + 1e-8) * 0.95
    return output

def process_channel_in_chunks(x, fs, chunk_duration=30.0, label="Channel"):
    chunk_size = int(chunk_duration * fs)
    total_samples = len(x)
    processed = []

    for i in tqdm(range(0, total_samples, chunk_size), desc=f"Processing {label}"):
        chunk = x[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        y = pitch_modulated_noise(chunk, fs)
        processed.append(y[:len(chunk)])

    return np.concatenate(processed)[:total_samples]

def process_stereo_file(input_path, output_path, fs=16000, chunk_duration=30.0):
    x, sr = librosa.load(input_path, sr=fs, mono=False)
    if x.ndim != 2:
        raise ValueError("Expected stereo audio.")

    left, right = x
    left_out = process_channel_in_chunks(left, fs, chunk_duration, label="Left")
    right_out = process_channel_in_chunks(right, fs, chunk_duration, label="Right")

    stereo_out = np.stack([left_out, right_out], axis=0)
    sf.write(output_path, stereo_out.T, fs)

# Example usage
if __name__ == "__main__":
    process_stereo_file(
        input_path = "/home/russelsa@ad.mee.tcd.ie/github/turntaking_master/turn-taking-projects/corpora/candor/candor_wav/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.wav",
        output_path = "/data/ssd3/russelsa/candor_noise_corpus/pitch/0a0cf5b9-84f6-4d8d-8001-ec7fd4b7437a.wav",
        fs=16000,
        chunk_duration=30.0
    )
