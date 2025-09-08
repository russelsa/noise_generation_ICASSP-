"""
    Runs faster-whisper on CANDOR stereo audio,
    generates plaintext transcript with one section per channel.
"""
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use GPU 0 only
import os
import pandas as pd
from datetime import timedelta
from pydub import AudioSegment
import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline

def run(fname, out_path, batchsize):

    model_size = "medium.en"
    device = "cuda"
    model_size = "medium.en"
    compute_type = "float16"

    model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)

    # --- Helper to format timestamps ---
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    # --- Process each session ---
    session_id = os.path.splitext(os.path.basename(fname))[0]

    if not fname.endswith(".wav"):
        return

    print(f"Processing session: {session_id}")

    audio = AudioSegment.from_wav(fname)
    output_lines = []

    for ch_index, ch_label in enumerate(["L", "R"]):

        tier_name = f"{session_id}--{ch_index}"

        # Extract mono channel and save temp
        temp_path = f"temp_{session_id}_{ch_label}.wav"
        audio.split_to_mono()[ch_index].export(temp_path, format="wav")

        # Transcribe
        segments, _ = batched_model.transcribe(temp_path, beam_size=15, batch_size=batchsize)

        # Add header and transcript
        output_lines.append(f"===== {tier_name} =====")
        for seg in segments:
            start = format_time(seg.start)
            text = seg.text.strip()
            output_lines.append(f"[{start}] {text}")
        output_lines.append("")  # blank line

        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))



if __name__ == "__main__":

    run(
        "sample_data/clean_audio/trimmed.wav",
        "sample_data/clean_audio/trimmed.txt",
        5
    )