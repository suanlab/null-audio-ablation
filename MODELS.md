# Checkpoints and upstream code

The exact third-party revisions the reported cells were produced with, recorded on 2026-11-09 from the local Hugging Face cache and the local checkouts. No weights are redistributed here; download each from its own release at the revision given.

## Model checkpoints (Hugging Face)

| System | Repository | Revision |
|---|---|---|
| Qwen2.5-Omni (7B) | `Qwen/Qwen2.5-Omni-7B` | `ae9e1690543ffd5c0221dc27f79834d0294cba00` |
| VideoLLaMA2.1-AV (7B) | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` | `d944d4298cb0188c3d5406eff42929cb67731361` |
| video-SALMONN 2+ (7B) | `tsinghua-ee/video-SALMONN2_plus_7B_full` | `aab28bd55258742f7dbaca319b2031f69b662f8f` |
| Whisper-large-v3 (probe; weight check) | `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` |

The BEATs checkpoint used by the encoder probe is `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` from the video-SALMONN v1 release, SHA-256 `e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c`.

## Upstream code

| Checkout | Repository | Commit | Local changes |
|---|---|---|---|
| `$VIDEOLLAMA2_DIR` | `github.com/DAMO-NLP-SG/VideoLLaMA2`, branch `audio_visual` | `8cedfbda23e779246112cb963416b95415448a91` | none to tracked files; our adapter `external/videollama2/eval_avqa_5mode.py` is copied into the checkout root |
| `$WORK_DIR/video-SALMONN-2` | `github.com/bytedance/video-SALMONN-2` | `61faec00abe4efb5c674c2ac154deb6c11cf5a91` | none; imported by `scripts/eval_avqa_videosalmonn2.py` |

## Environments

`requirements-frozen.txt` is the analysis environment (statistics, figures, tests, the Qwen2.5-Omni adapter, the probes). `envs/videollama2.txt`, `envs/videosalmonn2.txt` and `envs/qwenomni.txt` hold the exact package versions of the other three. All four are `pip list --format=freeze` output, so they pin versions, not wheels or build variants. CUDA builds are noted where pip records them (for example `torch==2.10.0+cu128`).
