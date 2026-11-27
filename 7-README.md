# Voice Cloning for Assistive Communication
### A Five-Stage Deep Learning Pipeline for Text-to-Speech

**Course:** Generative Artificial Intelligence (GenAI) — Practical Submission
**Department:** Computer Engineering, MIT Academy of Engineering, Alandi (D), Pune
**Guide:** Dr. Savita Mane

| Group Member | Roll No. (Exam Seat No.) |
| --- | --- |
| Samarth Kothale | 202301040229 |
| Devesh Kumar | 202301040241 |
| Abhishek Patil | 202301040232 |
| Parth Pawar | 202301040269 |

- **GitHub repository:** https://github.com/NexTechANKARA/synthetic-voice-cloning-tts
- **Live web app:** https://voicecloner-tts.netlify.app
- **Presentation (Canva):** https://www.canva.com/design/DAHbtlgxMfA/edit  ·  short link: https://canva.link/unz2zlo2shzqoqx

---

## 1. Project Overview

Voice Cloner is a text-to-speech (TTS) system with voice cloning, built from **five small generative
models trained from scratch** on 80 speakers of the VCTK corpus. Type a sentence, pick one of 80 built-in
voices (or clone a new voice from a short recording), and the system produces speech. The whole pipeline is
exported to ONNX (33.3 MB) and runs **entirely in the browser** — no audio ever leaves the device.

```
text ──► 3 FastSpeech-style Transformer ──► 4 Diffusion refiner ──► 5 GAN vocoder ──► audio
              (timing + spectrogram)          (shallow, strength 0.1)   (spectrogram to audio)
                     ▲
voice clip ──► 1 Denoising autoencoder ──► 2 Speaker VAE ──► speaker embedding (64 numbers)
```

| Model | Role | Params | Checkpoint |
| --- | --- | --- | --- |
| 1. Denoising Autoencoder | Cleans the reference voice clip | 0.047 M | `checkpoints/ae_best.pt` |
| 2. Speaker VAE | Turns a voice into a 64-D embedding (98.6% speaker accuracy) | 2.028 M | `checkpoints/vae_best.pt` |
| 3. FastSpeech-style Transformer | Predicts per-character duration and builds the mel | 6.804 M | `checkpoints/fastspeech_best.pt` |
| 4. Diffusion Refiner | Lightly sharpens the spectrogram (shallow, s=0.1) | 0.287 M | `checkpoints/diffusion_best.pt` |
| 5. GAN Vocoder | Turns the spectrogram into a 22.05 kHz waveform | 0.558 M | `checkpoints/gan_best.pt` |

## 2. Headline Results

| Result (7 test sentences, Whisper `small`) | Word error rate ↓ | Sentences perfect ↑ | Voice similarity ↑ |
| --- | --- | --- | --- |
| Original design (fixed-slot Transformer) | 0.960 | 0% | 0.629 |
| **Final pipeline (FastSpeech, duration-based)** | **0.083** | **71%** | **0.744** |

- Unseen (held-out) speakers: similarity 0.58, WER 0.083.
- Total size 9.72 M params / 33.3 MB ONNX; 9–60× faster than real time on 2 CPU cores.

## 3. Setup & Installation

Requires **Python 3.11** and PyTorch (GPU optional for inference, recommended for training).

```bash
# 1. Clone the repository
git clone https://github.com/NexTechANKARA/synthetic-voice-cloning-tts
cd synthetic-voice-cloning-tts

# 2. Install dependencies
pip install torch torchaudio librosa soundfile num2words
# for training / evaluation also:
pip install openai-whisper matplotlib tqdm
```

## 4. Usage

```python
import sys; sys.path.insert(0, ".")          # run from the repository folder
from vc_lib.infer import VoicePipeline
import soundfile as sf

pipe = VoicePipeline(".")                     # loads the five models
speaker = pipe.meta["train_speakers"][0]      # a built-in voice, e.g. "p225"
audio = pipe.synthesize("The meeting has been moved to Thursday afternoon.", speaker=speaker)
sf.write("out.wav", audio, 22050)

# clone a voice from a 5 to 20 second recording
emb = pipe.embed_reference(["my_voice.wav"])
sf.write("mine.wav", pipe.synthesize("Hello, this is my voice.", speaker_emb=emb), 22050)
```

`synthesize` also accepts `pace` (below 1 speaks faster, above 1 slower).

**End-to-end smoke test** (runs on synthetic data, no dataset needed):

```bash
python tests/smoke_test.py
```

## 5. Repository Layout

| Path | Contents |
| --- | --- |
| `vc_lib/` | Models, data preparation, training and inference code |
| `notebooks/` | Training and evaluation notebooks, run in order |
| `tools/export_onnx.py` | Export all five models to ONNX for the browser |
| `tests/smoke_test.py` | End-to-end test on synthetic data |
| `data/meta.json`, `data/speakers.json` | Settings and speaker list the models need |
| `checkpoints/` | Trained model weights |
| `backend/` | Web app API (Hugging Face Space) |
| `frontend/` | Web app website (Netlify) |

## 6. Training from Scratch

Training data (~30 GB) is not in the repository. Run the notebooks in order; each sets `ROOT` first.

| Notebook | Step |
| --- | --- |
| `00_data_prep` | Download VCTK, cut phrases with Whisper word timings, build the mel cache |
| `01_autoencoder` | Model 1 — Denoising Autoencoder |
| `02_speaker_vae` | Model 2 — Speaker VAE |
| `03b_fastspeech` | Model 3 — FastSpeech (final); `03_transformer` is the original design |
| `04b_diffusion` | Model 4 — Diffusion on FastSpeech output; `04_diffusion` is the original |
| `05_gan_vocoder` | Model 5 — GAN Vocoder |
| `06b_demo_pipeline` | Listen to and evaluate the full pipeline |
| `07_shallow_diffusion` | Diffusion strength study |
| `08_diffusion_predicted` | Diffusion retrained on predicted durations (negative result) |

## 6a. Files in this submission

| File | Content |
| --- | --- |
| `1-Voice_Cloning_Generative_AI_Practical_PPT_compressed.pdf` | Presentation (29 slides) |
| `2-GitHub_Repo_Link.pdf` | GitHub repository link |
| `0-Data_Preparation.ipynb` | Data preparation (VCTK, 80 speakers) |
| `3-Autoencoder_Voice_Cloning_Training.ipynb` | Model 1 — Denoising Autoencoder |
| `4-Speaker_VAE_Voice_Cloning_Training.ipynb` | Model 2 — Speaker VAE |
| `5-Transformer_Text_to_Mel_Training.ipynb` | Model 3 baseline — Transformer |
| `6-FastSpeech_Transformer_Training.ipynb` | Model 3 final — FastSpeech-style Transformer |
| `7-README.md` | This file |
| `Dataset_Link.pdf` | Dataset source and statistics |
| `Model_Files_Link.pdf` | Saved checkpoints and share links |
| `Report_of_Voice_Cloner.pdf` | Project report |

## 7. Ethics, Bias & Safety

- Research prototype, not a deployed product. All speaker data is from the open-access VCTK corpus.
- The app asks the user to confirm the voice is theirs or that they have permission before any clip is used.
- Synthetic speech should always be disclosed to listeners. Watermarking and a detector for the system's
  own output are planned safeguards before any public release.
- **Clone only voices you own or have explicit permission to use.**

## 8. License / Academic Use

Submitted as coursework for the Generative AI course, MIT Academy of Engineering. VCTK corpus is used
under its CC-BY-4.0 license.
