import os
import numpy as np
from scipy.io import wavfile
import tqdm

dir1 = "/home/russelsa@ad.mee.tcd.ie/github/turntaking_master/turn-taking-projects/corpora/candor/candor_wav"  # ← Change this to your actual path
dir2 = "/data/ssd4/russelsa/candor_speech_manipulations/intensity"  

files = os.listdir(dir1)
for filename in tqdm.tqdm(files):
    if filename.lower().endswith(".wav"):
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)

        if not os.path.exists(path2):
            print(f"Skipping {filename}: not found in dir2")
            continue

        # Load both files
        sr1, data1 = wavfile.read(path1)

        try:
            sr2, data2 = wavfile.read(path2)
        except Exception as e:
            print(e, "with", filename)
            print(filename)
            exit()

        # Check stereo format
        if data1.ndim != 2 or data2.ndim != 2 or data1.shape[1] != 2 or data2.shape[1] != 2:
            print(f"Skipping {filename}: not stereo")
            continue

        # Check sample rates
        if sr1 != sr2:
            print(f"Skipping {filename}: sample rate mismatch ({sr1} != {sr2})")
            continue

        len1 = data1.shape[0]
        len2 = data2.shape[0]

        if len2 < len1:
            pad = np.zeros((len1 - len2, 2), dtype=data2.dtype)
            data2 = np.vstack([data2, pad])
            print(f"Padded {filename} from {len2} → {len1} samples")
        elif len2 > len1:
            data2 = data2[:len1, :]
            print(f"Trimmed {filename} from {len2} → {len1} samples")
        else:
            print(f"{filename} already matches length")

        # Overwrite file in dir2
        wavfile.write(path2, sr2, data2)