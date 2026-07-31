import os
import sys
import io
import random
import warnings
import platform

# Suppress specific noisy warnings from Coqui TTS and PyTorch.
# (raw strings used below so the regex escapes are interpreted correctly
#  instead of triggering "invalid escape sequence" warnings on newer Python)
warnings.filterwarnings("ignore", category=UserWarning, message=r".*stft with return_complex=False.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.autocast.*")

# Force UTF-8 stdout so Tamil text prints correctly on Windows consoles.
# Guarded because sys.stdout doesn't always expose .buffer (e.g. some IDEs,
# redirected/piped output, or when stdout has already been wrapped).
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except AttributeError:
    pass

import argparse
import torch

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import soundfile as sf
import subprocess
import trainer.generic_utils
import trainer.trainer as trainer_module
import TTS.tts.layers.vits.transforms as vits_transforms


# Monkey-patch to prevent WinError 32 (file-in-use) from silently swallowing
# the real crash log when the trainer tries to delete an old experiment folder.
def safe_remove(experiment_path):
    print(f"Skipping deletion of {experiment_path} to preserve logs.")


trainer.generic_utils.remove_experiment_folder = safe_remove


# Monkey-patch to fix a StopIteration crash: the trainer library asks `git
# branch` for the current branch name (purely for run-metadata logging) and
# assumes at least one line starts with "*". That assumption breaks if the
# folder isn't a git repo, git isn't installed, or the repo has no commits
# yet — none of which should ever stop training. Fall back to "unknown"
# instead of crashing.
def safe_get_git_branch():
    try:
        out = subprocess.check_output(
            ["git", "branch"], stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
        current = next((line for line in out.split("\n") if line.startswith("*")), None)
        return current.replace("*", "").strip() if current else "unknown"
    except Exception:
        return "unknown"


trainer.generic_utils.get_git_branch = safe_get_git_branch
# Also patch the name inside trainer.trainer directly, since it may have been
# imported via `from trainer.generic_utils import get_git_branch`, in which
# case patching the source module alone wouldn't affect the already-bound name.
if hasattr(trainer_module, "get_git_branch"):
    trainer_module.get_git_branch = safe_get_git_branch


# Monkey-patch to fix a crash in the stochastic duration predictor's
# normalizing flow: "RuntimeError: min(): Expected reduction dim to be
# specified for input.numel() == 0". This happens when every value in a
# batch's flow input falls outside the spline's tail_bound (default ±5.0) —
# in practice this means the tensor has gone all-NaN/Inf from a numerically
# unstable step (fp16 mixed precision overflow is the most common cause with
# VITS's flow-based duration predictor). The library then calls torch.min()
# on the resulting empty tensor, which crashes instead of correctly treating
# "zero elements to transform" as a no-op. This patch only changes behavior
# for that empty-tensor edge case; normal (non-empty) calls are untouched.
_original_rational_quadratic_spline = vits_transforms.rational_quadratic_spline
_nan_spline_incidents = {"count": 0}


def safe_rational_quadratic_spline(inputs, *args, **kwargs):
    if inputs.numel() == 0:
        _nan_spline_incidents["count"] += 1
        n = _nan_spline_incidents["count"]
        if n in (1, 5, 20) or n % 100 == 0:
            print(f"⚠️  Numerically unstable batch skipped (incident #{n}) — the duration "
                  f"predictor's flow input went all-NaN/Inf, likely fp16 overflow. Training "
                  f"continues, but if this keeps recurring, restart with --no_amp for a "
                  f"more stable (if slightly slower) run.")
        return inputs.new_zeros(inputs.shape), inputs.new_zeros(inputs.shape)
    return _original_rational_quadratic_spline(inputs, *args, **kwargs)


vits_transforms.rational_quadratic_spline = safe_rational_quadratic_spline


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tamil VITS model.")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a previous experiment folder (containing config.json) to resume training from (restores model + optimizer + step count).")
    parser.add_argument("--restore", type=str, default=None,
                         help="Path to a specific .pth checkpoint to restore ONLY the model weights from (fresh optimizer/step count).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Override the training output directory.")
    parser.add_argument("--metadata", type=str, default="metadata.csv",
                         help="Name of the metadata file in the dataset directory (e.g. metadata.csv or metadata_phonemes.csv).")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Training batch size. Lower this (e.g. 8 or 4) if you hit an out-of-memory error.")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                         help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=1000,
                         help="Total number of training epochs.")
    parser.add_argument("--num_loader_workers", type=int, default=None,
                         help="Dataloader worker processes. Defaults to 0 on Windows (required to avoid WinError 32) and 4 elsewhere.")
    parser.add_argument("--no_amp", action="store_true",
                         help="Disable mixed-precision training even if a GPU is available.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.resume and args.restore:
        parser.error("--resume and --restore are mutually exclusive: --resume continues a full run "
                      "(model+optimizer+step count), --restore only loads model weights into a fresh run.")

    return args


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_metadata_to_ljspeech(original_meta: str, ljspeech_meta: str) -> int:
    """Convert `id|text` metadata.csv into Coqui's LJSpeech `id|text|normalized_text` format.

    Returns the number of valid lines written. Raises FileNotFoundError if the
    source file is missing, and ValueError if no valid lines were found.
    """
    if not os.path.isfile(original_meta):
        raise FileNotFoundError(
            f"Could not find metadata.csv at: {original_meta}\n"
            f"Make sure your dataset folder contains a 'metadata.csv' file formatted as 'relative_audio_path|text'."
        )

    with open(original_meta, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid_lines = []
    skipped = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 2:
            skipped += 1
            continue
        rel_path, text = parts
        text = text.strip()
        if not text:
            skipped += 1
            continue
        basename = os.path.splitext(os.path.basename(rel_path))[0]
        valid_lines.append(f"{basename}|{text}|{text}\n")

    if not valid_lines:
        raise ValueError(
            f"No valid lines found in {original_meta}. Expected each line to look like "
            f"'relative/audio/path.wav|transcript text'."
        )

    with open(ljspeech_meta, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    if skipped:
        print(f"Warning: skipped {skipped} malformed/empty line(s) while converting metadata.")

    return len(valid_lines)


def compute_mel_frames(duration_sec: float, sample_rate: int, hop_length: int) -> int:
    """Approximate number of mel-spectrogram frames for a given audio duration."""
    return int(duration_sec * sample_rate / hop_length)


def filter_unalignable_samples(samples, tokenizer, sample_rate: int, hop_length: int, safety_margin: float = 1.15):
    """Drop samples whose audio is too short for their transcript's token length.

    VITS's Monotonic Alignment Search (MAS) requires at least one mel frame per
    input token *after* blank-token interspersion (i.e. len(tokenizer.text_to_ids(text)),
    which already accounts for add_blank if enabled). When a chunk's audio is too
    short for its text — e.g. a mis-cut VAD boundary or a Whisper transcript that's
    longer than the actual speech in the clip — the MAS backtracking runs past the
    start of the array and raises "IndexError: index -N is out of bounds", killing
    the entire training run outright. This checks that ratio up front so bad chunks
    get skipped instead of crashing training hours later.

    `safety_margin` adds a small buffer above the bare minimum (>1.0) since actual
    mel-frame counts can vary slightly by a frame or two depending on padding.
    """
    kept, dropped = [], []
    for s in samples:
        audio_path = s["audio_file"]
        try:
            token_len = len(tokenizer.text_to_ids(s["text"]))
        except Exception as e:
            dropped.append((audio_path, f"tokenizer failed on text: {e}"))
            continue
        try:
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
        except Exception as e:
            dropped.append((audio_path, f"could not read audio file: {e}"))
            continue
        mel_frames = compute_mel_frames(duration, sample_rate, hop_length)
        if mel_frames < token_len * safety_margin:
            dropped.append((
                audio_path,
                f"audio too short for transcript: ~{mel_frames} mel frames available, "
                f"needs >= {token_len} (x{safety_margin} margin) tokens"
            ))
            continue
        kept.append(s)
    return kept, dropped


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- 1. SET PATHS ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "tamil_vits_dataset")
    original_meta = os.path.join(DATASET_DIR, args.metadata)
    ljspeech_meta = os.path.join(DATASET_DIR, "metadata_ljspeech.csv")

    if args.output_dir:
        OUT_DIR = os.path.abspath(args.output_dir)
    elif platform.system() == "Windows":
        # Kept at C:\ root by default to evade OneDrive Desktop Syncing WinError 32 bugs.
        OUT_DIR = r"C:\vits_output"
    else:
        OUT_DIR = os.path.join(BASE_DIR, "vits_output")

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 2. HARDWARE CHECK ---
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda and not args.no_amp
    if use_cuda:
        print(f"CUDA device detected: {torch.cuda.get_device_name(0)} | mixed precision: {use_amp}")
    else:
        print("No CUDA device detected — training will run on CPU and will be very slow. "
              "Mixed precision has been disabled.")

    # --- 3. FORMAT DATASET TO LJSPEECH ---
    print("Converting metadata.csv into Coqui TTS LJSpeech format...")
    num_lines = convert_metadata_to_ljspeech(original_meta, ljspeech_meta)
    print(f"Created {ljspeech_meta} with {num_lines} valid audio chunks.")

    # --- 4. CONFIGURATION ---
    if args.num_loader_workers is not None:
        num_workers = args.num_loader_workers
    else:
        # MUST be 0 on Windows to prevent WinError 32 file locking; safe to
        # parallelize on Linux/macOS for a meaningful data-loading speedup.
        num_workers = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_ljspeech.csv",
        path=DATASET_DIR
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="tamil_vits_from_scratch",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=5,
        num_loader_workers=num_workers,
        num_eval_loader_workers=num_workers,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner=None,        # Leave Tamil text completely unaltered.
        use_phonemes=False,       # Bypasses the need for espeak-ng installations on Windows.
        phoneme_language="ta",
        compute_input_seq_cache=True,
        print_step=50,
        print_eval=False,
        mixed_precision=use_amp,
        test_sentences=[
            "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "இந்த இயந்திர கற்றல் மாதிரி மிகவும் நன்றாக வேலை செய்கிறது.",
            "புதிய தொழில்நுட்பத்தை கற்பது மகிழ்ச்சி தருகிறது."
        ],
        output_path=OUT_DIR,
        datasets=[dataset_config],
        save_step=2000,
        eval_split_size=0.01
    )

    # --- 5. EXTRACT UNIQUE CHARACTERS DYNAMICALLY ---
    print("Loading datasets and mapping dynamic phonetics...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
    )

    if not train_samples:
        raise RuntimeError(
            "No training samples were loaded. Double-check that the audio files referenced in "
            "metadata.csv actually exist under the dataset folder."
        )
    if not eval_samples:
        print("Warning: eval split produced 0 samples (dataset may be too small for "
              "eval_split_size=0.01). Evaluation metrics will be skipped/empty.")

    punctuations = "!'(),-.:;? \n"
    valid_chars = set()
    for s in train_samples + eval_samples:
        for c in s["text"]:
            if c not in punctuations:
                valid_chars.add(c)
    all_chars_str = "".join(sorted(valid_chars))
    print(f"Dataset unique characters automatically extracted ({len(all_chars_str)}): {all_chars_str}")

    config.characters = CharactersConfig(
        characters=all_chars_str,
        punctuations=punctuations,
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>"
    )

    # --- 6. INITIALIZE AUDIO PROCESSOR & TOKENIZER ---
    print("Initializing Audio Processor and Tokenizer...")
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # --- 6b. FILTER OUT SAMPLES THAT WOULD CRASH MAS ALIGNMENT ---
    # This is the fix for: "IndexError: index -N is out of bounds ... in maximum_path_numpy"
    print("Checking audio/text length ratios to catch chunks that would crash "
          "alignment training (this replaces the mid-run 'index out of bounds' crash)...")
    train_samples, train_dropped = filter_unalignable_samples(
        train_samples, tokenizer, config.audio.sample_rate, config.audio.hop_length
    )
    eval_samples, eval_dropped = filter_unalignable_samples(
        eval_samples, tokenizer, config.audio.sample_rate, config.audio.hop_length
    )
    all_dropped = train_dropped + eval_dropped
    if all_dropped:
        log_path = os.path.join(OUT_DIR, "dropped_samples.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for path, reason in all_dropped:
                f.write(f"{path}\t{reason}\n")
        print(f"Dropped {len(all_dropped)} sample(s) whose audio is too short for their "
              f"transcript (these would have crashed MAS alignment mid-training). "
              f"Full list with reasons written to: {log_path}\n"
              f"Worth spot-checking these in vits_extract.py's VAD chunking / Whisper "
              f"output — short/cut audio or hallucinated transcripts are the usual cause.")
    if not train_samples:
        raise RuntimeError(
            "All training samples were filtered out as unalignable. This points to a "
            "systematic problem in the dataset (e.g. audio consistently far too short "
            "for its transcripts) rather than a few bad chunks — check dropped_samples.log."
        )

    # --- 7. INITIALIZE MODEL AND START TRAINING ---
    print("Initializing VITS model...")
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # --- 8. BUILD TRAINER AND FIT ---
    trainer_args = TrainerArgs()
    trainer_args.use_cuda = use_cuda
    if args.resume:
        if not os.path.isdir(args.resume):
            raise FileNotFoundError(f"--resume path does not exist or is not a directory: {args.resume}")
        trainer_args.continue_path = args.resume
    if args.restore:
        if not os.path.isfile(args.restore):
            raise FileNotFoundError(f"--restore checkpoint does not exist: {args.restore}")
        trainer_args.restore_path = args.restore

    trainer = Trainer(
        trainer_args,
        config,
        OUT_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\n🚀 Starting VITS Training! (Watch the console for progress logs).")
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. The last checkpoint saved under "
              f"{OUT_DIR} can be resumed with --resume.")


if __name__ == "__main__":
    main()