import os
import torchaudio 
import torch
from copy import copy 
import numpy as np
import torchaudio.functional as F
from dataset_management.dataset_manager.src.audio_manager import AudioManager


def add_noise_to_file(input_file, output_file, noise_files, snr_dbs, repeat=False):

    speech, sr_speech = AudioManager.load_waveform(input_file, normalize=True, sample_rate=16000, channel_first=True)
    noise, sr_noise = AudioManager.load_waveform(np.random.choice(noise_files), normalize=True, mono=True, sample_rate=16000, channel_first=True)

    start_sample = 0

    noisy_speech = copy(speech)

    i=0
    while noise.shape[-1] < speech.shape[-1]:
        i+=1

        if i%10==0 or not repeat:
            n2, sr_noise = AudioManager.load_waveform(np.random.choice(noise_files), normalize=True, mono=True, sample_rate=16000, channel_first=True)
            # n2 = n2/n2.max()
        else:
            n2=noise
        
        noise = torch.cat((noise, n2), dim=-1)

    noise = noise[:, :speech.shape[-1]]
    noise_stereo = torch.concatenate((noise, noise), dim=0)

    noisy_speech = F.add_noise(speech, noise_stereo, torch.tensor([snr_dbs]))

    torchaudio.save(output_file, noisy_speech, sr_speech)


def get_noise(speech, noise_files, repeat, normalize, volume_norm=False):
    """
        repeatedly concatenated noise files together with or without repetiion until it is the same length as the speech
    """

    new_segment = np.random.choice(noise_files)
    new_segment, _ = AudioManager.load_waveform(new_segment, mono=True, normalize=normalize, sample_rate=16000, transpose=True)

    noise = copy(new_segment)

    i=0
    while noise.shape[-1] < speech.shape[-1]:

        if repeat and i%10 == 0:
            new_segment = noise

        else:
            new_segment = np.random.choice(noise_files)
            new_segment, _ = AudioManager.load_waveform(new_segment, mono=True, normalize=normalize, sample_rate=16000, transpose=True) 
            
            if volume_norm:
                new_segment = new_segment / new_segment.max()

        
        noise = torch.concat((noise, new_segment), dim=-1)
        i+=1
    noise = noise[:, :speech.shape[-1]]

    return noise


def apply_noise_to_channel(speech, speech_no_silences, noise, snr, mode):
    """
        apply noise to a mono audio channel
        speech: the speech 
        speech_no_silences: identical to speech but silences removed
        noise: a noise file the same length as speech
    """

    noise_clipped = copy(noise)
    if speech_no_silences.shape[-1] < noise.shape[-1]:
        noise_clipped = noise_clipped[..., :speech_no_silences.shape[-1]]

    speech = speech.numpy().astype(np.float32)
    speech_no_silences = speech_no_silences.numpy().astype(np.float32)
    noise = noise.numpy().astype(np.float32)
    noise_clipped = noise_clipped.numpy().astype(np.float32)

    clean_rms = np.sqrt(np.mean(np.square(speech_no_silences)))
    noise_rms = np.sqrt(np.mean(np.square(noise)))

    alpha = clean_rms / (noise_rms * (10**(snr/20)))

    scaled_noise = alpha * noise
    mixed = speech + scaled_noise
    
    scaled_noise = alpha * noise_clipped
    mixed_ns = speech_no_silences + scaled_noise

    mixed = mixed.squeeze()
    mixed = torch.tensor(mixed).unsqueeze(dim=0)

    if mixed.abs().max() > 1: 
        # print("normalising to prevent clipping")
        mixed = mixed / mixed.abs().max()

    mixed_ns = mixed_ns.squeeze()
    mixed_ns = torch.tensor(mixed_ns).unsqueeze(dim=0)

    if mixed_ns.abs().max() > 1: 
        # print("normalising to prevent clipping")
        mixed_ns = mixed_ns / mixed_ns.abs().max()

    speech = torch.tensor(speech)
    noise = torch.tensor(noise)
    speech_no_silences = torch.tensor(speech_no_silences)

    return mixed

    # # Compute RMS in 1-second chunks
    # num_secs = speech_no_silences.shape[-1] // fs
    # track_rms = []

    # for i in range(num_secs - 1):
    #     data = speech_no_silences[:, i * fs: int((i+0.2)*fs)]
    #     data_rms = torch.sqrt(torch.mean(data**2))
    #     track_rms.append(data_rms.item())

    # # Order RMS values starting with the highest
    # track_rms = sorted(track_rms, reverse=True)
    # cutoff = int(num_secs * rms_thresh)
    # # cutoff=1
    # use_rms = track_rms[:cutoff]

    # # Compute target alpha for higher power elements
    # alpha = np.mean(use_rms) / (rms_nse * (10**(snr/10)))
    # x_noisy_naomi = speech + (alpha * noise)

    # # Normalize the noisy signal
    # x_noisy_naomi_norm = x_noisy_naomi / torch.max(torch.abs(x_noisy_naomi))

    # return x_noisy_naomi_norm

   