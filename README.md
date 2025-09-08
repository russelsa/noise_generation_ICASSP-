## Code Accompanying ICASSP 2026 Submission 

_PROSODIC AND LEXICAL CUES IN TURN-TAKING WITH SELF-SUPERVISED SPEECH REPRESENTATIONS_
Sam O’Connor Russell, Delphine Charuau and Naomi Harte

### Sample data
Sample utterances of each manipulation are provided in `sample_utterances` to listen to. Alternaitvely the below code can be run to generate these noises from a file, provided an utterance-aligned transcript in TextGrid format is available. We have provided a sample in `sample_data/clean_audio` that works with the below code.

### requirements 
`librosa, pyworld, faster-whisper`

### Overview

Our paper involves the generation of described below for a sample session in `sample_data/clean_audio`

1) **generation of prosodic noise**
To generate noise which follows the pitch and intensity of speech, run `scripts/vocoder/pinknoise_vocoder.py` and see outputs in `sample_data/prosodic_noise`. Additional noises follow pitch only (with intenstity flattened to the utterance mean), and intensity only (pitch flattened to the utterance mean). 

2) **generation of prosodic manipulation**
To generate speech which preserves lexical information whilst flattening prosody, run `scripts/vocoder/prosodic_manipulation.py` and see outputs in `prosodic_manipulation`. Both pitch and intensity are flattened, or just pitch / intensity. 

3) running whisper 

4) WER intellegibility proxy 
run `scripts/WER_intellegibility/run_faster_whisper.py` to transcribe files and compute the WER relative to clean speech

5) SNR adding 
to add manipulations at various SNRs code is provided in our earlier work, see below. 

To hear sample utterances `sample_data/sample_utterance`

Spetrogram.jpg

### Background noise and models

For all model and background noise (babble, music and speech) code please see our earlier work `github.com/russelsa/mm-vap` from [1,2]

### References


[1] _Visual Cues Support Robust Turn-taking Prediction in Noise_ Sam O'Conor Russell and Naomi Harte, Proc. of INTERSPEECH 2025
```
@inproceedings{oconnorrussell25_interspeech,
  title     = {{Visual Cues Support Robust Turn-taking Prediction in Noise}},
  author    = {Sam {O'Connor Russell} and Naomi Harte},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1073--1077},
  doi       = {10.21437/Interspeech.2025-668},
  issn      = {2958-1796},
}
```


[2] _Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction_ Sam O'Conor Russell and Naomi Harte, ACL Findings 2025 
```
@inproceedings{russell-harte-2025-visual,
    title = "Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction",
    author = "Russell, Sam O{'}Connor  and
      Harte, Naomi",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.12/",
    doi = "10.18653/v1/2025.findings-acl.12",
    pages = "209--221",
}
```
# noise_generation_ICASSP-
