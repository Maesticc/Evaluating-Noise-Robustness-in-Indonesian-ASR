# Whisper checkpoint selection

Decision record for the Whisper checkpoint. The plan (v5) specifies Whisper Large-v3; section 7.3 allows a
smaller checkpoint only if the hardware requires it, and the change must be documented.

## Problem

The pilot (30 test utterances x 5 conditions = 150 rows, RTX 3050 Laptop 4 GB) ran
`openai/whisper-large-v3` in fp16 with batch size 1:

| Model | WER clean | WER 0 dB | RTFx | Peak GPU memory |
|---|---|---|---|---|
| Whisper large-v3 | 6.35 % | 21.74 % | 1.08 | 3.39 GB of 4.29 GB |
| Wav2Vec2 XLSR Indonesian | 30.10 % | 59.87 % | 22.64 | 2.69 GB of 4.29 GB |

Accuracy is not the problem on this laptop; throughput is. The full test set is 15,673 eligible utterances x 5
conditions = 78,365 rows, 94.7 h of audio (manifest durations), so large-v3 at RTFx 1.08 needs roughly
**88 GPU-hours** on this laptop (Wav2Vec2: about 4 h). RTFx depends on other load on the laptop: an earlier
pilot run measured 0.85 for large-v3 (about 111 h), so timings are taken with no other GPU-heavy apps open.
Batch 4 does not fit in 4 GB (4.14 GB, spills into shared memory, 2-3x slower).

## Candidates if Large-v3 cannot be run

| Checkpoint | Indonesian accuracy in noise | Memory (fp16 weights) | Speed | Verdict |
|---|---|---|---|---|
| `whisper-large-v3` (1.55B) | Best measured (pilot above) | ~3.1 GB, batch 1 only on 4 GB | Slowest: 32 decoder layers | **Planned model** |
| `whisper-large-v3-turbo` (809M) | Expected close to large-v3 (same encoder) | ~1.6 GB | Several times faster: 4 decoder layers | Fallback if required |
| `whisper-medium` (769M) | Below large-v3 | ~1.5 GB | Slower than turbo: 24 decoder layers | Dominated by turbo |
| `whisper-small` (244M) | Clearly lower, worst at low SNR | < 1 GB | Fast | Not a fair "best Whisper" |
| `distil-large-v3` | - | - | - | English-only, not usable |
| faster-whisper / CTranslate2 | Near-identical to its source checkpoint | Lower (int8) | 2-4x faster | Different runtime than the other models; RTFx not comparable |
| Community Indonesian fine-tunes | Possibly higher | - | - | Violates "pretrained, no fine-tuning"; training data may overlap the corpus |

Why turbo is the fallback: its encoder, where most noise robustness comes from, is the same as
large-v3; only the decoder is pruned (32 to 4 layers). This is a **hypothesis** to check before use.

## Local vs. cloud

- **Hosted APIs** (OpenAI, Groq, ...): not used. The served model version is opaque and can change,
  decoding cannot be fully controlled, and network latency makes RTFx incomparable with the local models.
- **Cloud or larger GPU** (e.g. Kaggle / Colab T4 16 GB, or a local 8 GB GPU): runs Large-v3 as planned with
  the same code. All three models are then run on that hardware, so RTFx stays comparable.

## Decision rule

1. **Keep Large-v3** (the plan's model) whenever a GPU is available on which the full run is feasible.
2. **Only if no such GPU is available**, switch to turbo (plan 7.3). Check the accuracy change first on a
   **validation development subset** (val speakers + `noise_train` noise of the six evaluation categories),
   never on test results (plan 9.4, 11.1):
   ```bash
   python -m src.noise_mixing build-pairs --split val --pilot
   python -m src.inference --split val --pilot --models whisper            # Large-v3
   # keep the Large-v3 run under its own tag
   mv output/predictions/predictions_val_pilot_whisper.csv output/predictions/predictions_val_pilot_largev3_whisper.csv
   mv output/predictions/run_val_pilot_whisper.json output/predictions/run_val_pilot_largev3_whisper.json
   python -m src.evaluate --tag val_pilot_largev3
   # set models.whisper.checkpoint to openai/whisper-large-v3-turbo in config.yaml, then
   python -m src.inference --split val --pilot --models whisper            # Turbo
   python -m src.evaluate --split val --pilot
   ```
   Compare `results_val_pilot_largev3/` with `results_val_pilot/`. Use turbo only if its WER lies inside
   Large-v3's 95 % bootstrap CI in every condition (0 dB in particular).
3. The comparison keeps three systems with one Whisper checkpoint (plan 9.1). Record the outcome, RTFx and
   peak memory here and in the README, and freeze `config.yaml` before the full run (plan 14, step 10).

## Outcome

**Large-v3 kept (rule 1).** With the lecturer's approval (2026-11-23), the full run was done on Kaggle on one
NVIDIA Tesla T4 (16 GB), with all three models on the same GPU. The turbo fallback was not needed, so the
validation comparison above was not run.

| | Laptop pilot (RTX 3050, 4 GB) | Kaggle full test (T4, 16 GB) |
|---|---|---|
| Checkpoint | `openai/whisper-large-v3` @ `06f233f`, fp16 | same |
| Batch size | 1 | 16 |
| RTFx | 1.08 | 17.8 |
| Peak GPU memory | 3.39 GB | 7.73 GB |
| Time | 150 rows | 78,365 rows (94.7 h of audio) in 5.3 h |

- The batch size only changes speed: the pilot repeated on the T4 with batch 16 gave transcripts identical to the
  laptop run with batch 1 (150/150, `output/kaggle/PILOT_CHECK.json`).
- `config.yaml` keeps `batch_size: 1` for the laptop; the Kaggle setting is saved in
  `output/kaggle/config_kaggle_test.yaml`.
- Full-test WER: 4.79 % clean, 21.94 % at 0 dB, the best of the three models in every condition
  (see the README and `report/results_analysis.md`).
