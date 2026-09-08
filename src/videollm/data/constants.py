"""Constants for VideoLLM: special tokens, default paths, and modal indices."""

# Special tokens
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_AV_TOKEN = "<av>"  # Audio-visual joint token

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_AUDIO_PATCH_TOKEN = "<aud_patch>"

IGNORE_INDEX = -100

# Modal index mapping for tokenizer embedding expansion
IMAGE_TOKEN_INDEX = -200
VIDEO_TOKEN_INDEX = -201
AUDIO_TOKEN_INDEX = -202
AV_TOKEN_INDEX = -203

MODAL_INDEX_MAP: dict[str, int] = {
    DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN: VIDEO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN: AUDIO_TOKEN_INDEX,
    DEFAULT_AV_TOKEN: AV_TOKEN_INDEX,
}

# Default model identifiers
DEFAULT_VISION_ENCODER = "google/siglip-so400m-patch14-384"
DEFAULT_AUDIO_ENCODER = "laion/larger_clap_general"
DEFAULT_LLM = "Qwen/Qwen2-7B-Instruct"

# Video processing defaults
DEFAULT_NUM_FRAMES = 16
DEFAULT_IMAGE_SIZE = 384
DEFAULT_VIDEO_FPS = 1  # frames per second for sampling
MAX_NUM_FRAMES = 128

# Audio processing defaults
DEFAULT_AUDIO_SAMPLE_RATE = 48000
DEFAULT_AUDIO_DURATION = 10.0  # seconds
DEFAULT_AUDIO_MEL_BINS = 64
