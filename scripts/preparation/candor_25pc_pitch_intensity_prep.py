from tqdm import tqdm
import os
import shutil
import random
from pathlib import Path
from noise_generation.scripts.candor_noise_test_set_prep import add_noise_to_file_stereo

# Define directories
dir1 = Path("/home/russelsa@ad.mee.tcd.ie/github/turntaking_master/turn-taking-projects/corpora/candor/candor_wav")
augmentA = Path("/data/ssd3/russelsa/candor_noise_corpus/intensity")  # Replace with actual path
augmentB = Path("/data/ssd3/russelsa/candor_noise_corpus/pitch")  # Replace with actual path
augmentC = Path("/data/ssd3/russelsa/candor_noise_corpus/pitch_and_intensity")  # Replace with actual path
output = Path("/data/ssd2/russelsa/candor_augment_25pc_noise_pitch_intensity")      # Replace with actual path

# DON'T do this -- same files same silences
candor_no_sil = dir1

# Make sure output directory exists
output.mkdir(parents=True, exist_ok=True)

# For logging
log_path = output / "copy_log.csv"
log_lines = ["filename,source\n"]

# List of augment directories
augment_dirs = [augmentA, augmentB, augmentC]

wavs = dir1.glob("*.wav")

# Iterate over all .wav files in dir1
for wav_path in tqdm(wavs):
    filename = wav_path.name

    if random.random() < 0.75:
        # 75% probability: copy original
        src = wav_path
        source_label = "original"

        # Copy to output
        dst = os.path.join(output, "wavs" , filename)
        shutil.copy(src, dst)
        print(f"shutil.copy({src}, {dst})")

    else:
        
        # 25% probability: pick random augment
        aug_dir = random.choice(augment_dirs)
        src = os.path.join(aug_dir, filename)
        dsr = os.path.join(output, "wavs", filename)

        if not os.path.exists(src):
            print(f"Warning: Augmented file not found for {filename} in {aug_dir}")
            continue

        source_label = aug_dir.name

        # mix noise with the signal at 25pc SNR
        add_noise_to_file_stereo(
            input_file=str(wav_path),
            no_silences_directory=str(candor_no_sil),
            output_file=str(dsr),
            noise_files=[str(src)], 
            snr_dbs=0
        )
        print(f"add_noise_to_file_stereo(input_file{wav_path},no_silences_directory={candor_no_sil},output_file={dsr},noise_files={[src]},snr_dbs=0)")


    # Log source
    log_lines.append(f"{filename},{source_label}\n")

    # Save log
    with open(log_path, "w") as f:
        f.writelines(log_lines)

print(f"Done. Files copied to: {output}")
print(f"Log saved to: {log_path}")
