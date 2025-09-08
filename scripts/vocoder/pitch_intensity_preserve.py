import os
import glob
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from textgrid import TextGrid
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def match_rms(y, target):
    rms_y = np.sqrt(np.mean(y ** 2)) + 1e-8
    rms_target = np.sqrt(np.mean(target ** 2)) + 1e-8
    return y * (rms_target / rms_y)

def fast_amplitude_envelope(x, frame_size=1024):
    return np.sqrt(np.convolve(x**2, np.ones(frame_size)/frame_size, mode='same'))

def flatten_pitch(x, fs=16000):
    x = np.ascontiguousarray(x, dtype=np.float64)
    frame_period = 10.0
    fft_size = 512
    f0, timeaxis = pw.harvest(x, fs, frame_period=frame_period)
    sp = pw.cheaptrick(x, f0, timeaxis, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, timeaxis, fs, fft_size=fft_size)

    voiced_f0 = f0[f0 > 0]
    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 100.0
    f0_flat = np.where(f0 > 0, mean_f0, 0.0)

    y_f0_flat = pw.synthesize(f0_flat, sp, ap, fs, frame_period)
    def match_length(sig, target_len):
        return np.pad(sig, (0, max(0, target_len - len(sig))))[:target_len]
    y_f0_flat = match_length(y_f0_flat, len(x))

    # 3. Flatten intensity
    # Calculate RMS energy for each frame and normalize all frames to same energy
    log_energy = np.log(np.sum(sp + 1e-8, axis=1))  # Log-energy of spectral envelope
    mean_log_energy = np.mean(log_energy)
    energy_scale = np.exp(mean_log_energy - log_energy)

    sp_flat = sp * energy_scale[:, np.newaxis]

    # 4. WORLD synthesis
    y_intensity_flat = pw.synthesize(f0, sp_flat, ap, fs, frame_period)
    y_f0_intensity_flat = pw.synthesize(f0_flat, sp_flat, ap, fs, frame_period)

    y_intensity_flat = match_length(y_intensity_flat, len(x))
    y_f0_intensity_flat = match_length(y_f0_intensity_flat, len(x))

    return y_f0_flat, y_intensity_flat, y_f0_intensity_flat

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
    y_f0_flat, y_intensity_flat, y_f0_intensity_flat = flatten_pitch(segment, fs)
    
    def tidy(x, envelope=envelope):
        min_len = min(len(envelope), len(x))
        envelope = envelope[:min_len]
        
        x = x[:min_len]
        x = x * envelope
        
        ref = segment[:min_len]
        return match_rms(x, ref)
    
    y_f0_flat, y_f0_intensity_flat, y_intensity_flat = tidy(y_f0_flat), y_f0_intensity_flat, y_intensity_flat
    
    return start, y_f0_flat, y_f0_intensity_flat, y_intensity_flat

def process_channel(wav_channel, intervals, fs, channel_name=""):
    total_samples = int(intervals[-1].maxTime * fs)
    pitch_output = np.zeros(total_samples)
    shaped_output = np.zeros(total_samples)
    intensity_output = np.zeros(total_samples)

    args = [(wav_channel, interval, fs) for interval in intervals if interval.mark.strip() != ""]

    for i, interval in enumerate(tqdm(args, desc=f"Segments ({channel_name})", leave=False)):
        if i > 10:
            return pitch_output, shaped_output, intensity_output
        result = process_interval(interval)
        if result is None:
            continue
        start, y_pitch, y_shaped, y_intensity = result
        end = start + len(y_pitch)
        pitch_output[start:end] = y_pitch[:end - start]
        shaped_output[start:end] = y_shaped[:end - start]
        intensity_output[start:end] = y_intensity[:end - start]

    return pitch_output, shaped_output, intensity_output

def manipulations_textrid(input_wav, input_textgrid, output_pitch, output_shaped, output_intensity, fs=16000):
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

    if not os.path.exists(tg_file):
        print(f"❌ Skipping {base}: No TextGrid")
        return
    # if os.path.exists(out_pitch) and os.path.exists(out_shaped) and os.path.exists(out_intensity):
    #     print(f"⏭ Skipping {base}: Already processed")
    #     return

    print(f"🔄 Processing {base}")
    # try:
    manipulations_textrid(
        input_wav=wav_file,
        input_textgrid=tg_file,
        output_pitch=out_pitch,
        output_shaped=out_shaped,
        output_intensity=out_intensity,
        fs=16000
    )
    print(f"✅ Done: {base}")
    # except Exception as e:
    #     print(f"❌ Error processing {base}: {e}")

if __name__ == "__main__":
    input_dir = "/home/russelsa@ad.mee.tcd.ie/github/turntaking_master/turn-taking-projects/corpora/candor/candor_wav"
    tg_dir = "/home/russelsa@ad.mee.tcd.ie/github/turntaking_master/turn-taking-projects/corpora/candor/candor_ipu"
    out_dir_pitch = "/data/ssd2/russelsa/candor_speech_manipulations/pitch"
    out_dir_shaped = "/data/ssd2/russelsa/candor_speech_manipulations/pitch_and_intensity"
    out_dir_intensity = "/data/ssd2/russelsa/candor_speech_manipulations/intensity"

    os.makedirs(out_dir_pitch, exist_ok=True)
    os.makedirs(out_dir_shaped, exist_ok=True)
    os.makedirs(out_dir_intensity, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    print(f"🔎 Found {len(wav_files)} .wav files")

    # 🐢 Serial (easier to debug, shows progress)
    for wav_file in tqdm(wav_files[:5], desc="Processing files"):
        process_one_file(wav_file, input_dir, tg_dir, out_dir_pitch, out_dir_shaped, out_dir_intensity)
        exit()

    # # 🚀 To enable multiprocessing later:
    # with Pool(cpu_count()) as pool:
    #     pool.starmap(process_one_file, [(f, input_dir, tg_dir, out_dir_pitch, out_dir_shaped, out_dir_intensity) for f in wav_files])
