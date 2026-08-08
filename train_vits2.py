"""Prepare this dataset and launch a real VITS2 training backend.

Coqui TTS 0.22 ships VITS, but not VITS2. This runner uses the VITS2
implementation from https://github.com/p0p4k/vits2_pytorch and turns this
project's ``wavs/path.wav|ipa text`` metadata into its train/validation lists.

First-time setup (run once):
    python train_vits2.py --install-backend

Then train:
    python train_vits2.py --metadata metadata_phonemes_zipa_no_diacritics.csv
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "veenv" / "Scripts" / "python.exe"
BACKEND_DIR = PROJECT_DIR / "vits2_pytorch"
BACKEND_URL = "https://github.com/p0p4k/vits2_pytorch.git"
MODEL_NAME = "tamil_vits2"


def restart_in_project_venv():
    """Use the project venv before importing/running backend dependencies."""
    if not VENV_PYTHON.is_file():
        return
    try:
        if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
            return
    except OSError:
        return
    result = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and train Tamil VITS2.")
    parser.add_argument(
        "--dataset_dir",
        default=str(PROJECT_DIR / "tamil_vits_dataset"),
        help="Path to the training dataset directory.",
    )
    parser.add_argument(
        "--metadata",
        default="metadata_phonemes_zipa_no_diacritics.csv",
        help="Two-column metadata filename inside the dataset directory.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU training batch size.")
    parser.add_argument("--epochs", type=int, default=2000, help="Total training epochs.")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Checkpoint interval in steps.")
    parser.add_argument("--val_fraction", type=float, default=0.01, help="Validation split fraction.")
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes to patch into the VITS2 train.py (which "
            "hardcodes 8). Default 0 avoids Windows STATUS_ACCESS_VIOLATION crashes "
            "caused by nesting DataLoader worker spawns inside train.py's own "
            "mp.spawn() DDP process once CUDA is initialized. Raise this only if "
            "you've confirmed your setup tolerates nested multiprocessing."
        ),
    )
    parser.add_argument(
        "--disable_crash_diagnostics",
        action="store_true",
        help=(
            "Skip setting PYTHONFAULTHANDLER=1 and CUDA_LAUNCH_BLOCKING=1 on the "
            "training subprocess. These are on by default to turn opaque Windows "
            "access-violation crashes into an actual Python traceback / CUDA error "
            "message; CUDA_LAUNCH_BLOCKING adds overhead, so disable this once the "
            "crash is diagnosed and fixed."
        ),
    )
    parser.add_argument(
        "--install-backend",
        action="store_true",
        help="Clone VITS2 and install its Python requirements into this project venv.",
    )
    return parser.parse_args()


def install_backend():
    if not BACKEND_DIR.is_dir():
        print(f"Cloning VITS2 backend into {BACKEND_DIR}...")
        subprocess.run(["git", "clone", "--depth", "1", BACKEND_URL, str(BACKEND_DIR)], check=True)
    # The project environment already supplies PyTorch, NumPy, SciPy, librosa,
    # TensorBoard, and Matplotlib. Installing the upstream requirements file
    # would also replace onnxruntime-gpu with an old CPU-only onnxruntime build,
    # which would break ZIPA extraction. Only Cython is required to compile the
    # VITS2 alignment extension below.
    print("Installing the VITS2 alignment build dependency (Cython)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "Cython==3.0.2"], check=True)
    alignment_dir = BACKEND_DIR / "monotonic_align"
    print("Building VITS2 monotonic-alignment extension...")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=alignment_dir, check=True)


def patch_dataloader_workers(num_workers):
    """Rewrite the backend's hardcoded DataLoader num_workers.

    Upstream train.py hardcodes ``num_workers=8`` on both the train and eval
    DataLoaders. On Windows this nests a second multiprocessing spawn tree
    (DataLoader workers) inside the DDP rank process that train.py's own
    mp.spawn() already created -- and doing that once CUDA is initialized in
    the parent is a well-known source of STATUS_ACCESS_VIOLATION (exit code
    3221225477) crashes. Patching to num_workers=0 keeps all data loading on
    the main thread and avoids the nested spawn entirely. Idempotent: re-running
    with the same value is a no-op, and re-running with a different value
    re-patches from whatever value is currently on disk.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    original = train_py.read_text(encoding="utf-8")
    import re

    # Remove any existing patched flags to prevent duplicates on re-runs
    text = re.sub(r"persistent_workers=(?:True|False)\s*,\s*", "", original)
    text = re.sub(r"pin_memory=(?:True|False)\s*,\s*", "", text)

    persistent = "True" if num_workers > 0 else "False"
    patched, count = re.subn(
        r"num_workers=\d+,",
        f"num_workers={num_workers}, persistent_workers={persistent}, pin_memory=True,",
        text
    )
    if count == 0:
        print(f"Warning: no 'num_workers=<N>,' pattern found in {train_py.name}; skipping patch.")
        return
    if patched != original:
        train_py.write_text(patched, encoding="utf-8")
        print(f"Patched {train_py.name}: DataLoader num_workers -> {num_workers} ({count} occurrence(s)).")


def patch_windows_distributed_backend():
    """Swap the backend NCCL for the Windows-supported gloo backend.

    Upstream train.py hardcodes dist.init_process_group(backend="nccl", ...).
    NCCL is Linux/CUDA-only -- PyTorch's official Windows builds don't ship
    it. With a single GPU (world_size=1) init_process_group can still
    "succeed" since there's no real cross-process rendezvous needed yet, so
    training reaches the first step -- but the moment DDP's Reducer issues
    its first real collective op (the gradient allreduce right after the
    "find_unused_parameters" warning), it hits the missing/broken NCCL
    library and hard-crashes the process with STATUS_ACCESS_VIOLATION (exit
    code 3221225477 / -1073741819), not a catchable Python exception. gloo is
    the backend PyTorch documents as supported on Windows, including for CUDA
    tensors, so swap to it. Idempotent: no-op if already patched.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    original = train_py.read_text(encoding="utf-8")
    patched = original.replace('backend="nccl"', 'backend="gloo"')
    if patched != original:
        train_py.write_text(patched, encoding="utf-8")
        print(f"Patched {train_py.name}: distributed backend nccl -> gloo (Windows crash workaround).")


def patch_find_unused_parameters():
    """Disable DDP's find_unused_parameters=True.

    train.py wraps all three networks in DistributedDataParallel with
    find_unused_parameters=True, which makes DDP perform an extra traversal
    of the autograd graph on every backward() call to detect parameters that
    didn't receive gradients. train.py's own runtime warning confirms this
    project's model has zero unused parameters, so the flag buys nothing --
    and the crash traceback (Windows access violation inside
    torch.Tensor.backward(), no Python frame) points at exactly this code
    path. Disabling it removes that extra traversal without changing training
    behavior. Idempotent: no-op if already patched.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    original = train_py.read_text(encoding="utf-8")
    patched = original.replace("find_unused_parameters=True", "find_unused_parameters=False")
    if patched != original:
        print(f"Patched {train_py.name}: DDP find_unused_parameters -> False (Windows crash workaround).")
        train_py.write_text(patched, encoding="utf-8")


def patch_skip_ddp_single_gpu():
    """Bypass DistributedDataParallel entirely for single-GPU training.

    Two separate, independently-justified DDP workarounds (nccl -> gloo,
    find_unused_parameters=False) each shifted the Windows access-violation
    crash later in the batch loop but never eliminated it -- it consistently
    dies inside torch.Tensor.backward(), in native code with no Python frame,
    which is the signature of DDP's autograd-hook-triggered gradient
    synchronization. With a single GPU there's no actual multi-process work
    for DDP to coordinate in the first place (mp.spawn only ever creates one
    process here), so DDP is pure unnecessary surface area for this crash.
    Replace it with a trivial wrapper that exposes the same `.module`
    interface (which train.py and utils.py's checkpoint code both rely on)
    but performs no torch.distributed communication at all. Idempotent: safe
    to run repeatedly, and tolerant of the DDP call still saying either
    find_unused_parameters=True or =False depending on patch order.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    original = train_py.read_text(encoding="utf-8")
    text = original

    if "_SingleProcessModelWrapper" not in text:
        wrapper_class = (
            "\n\n"
            "class _SingleProcessModelWrapper(nn.Module):\n"
            '    """Drop-in substitute for DistributedDataParallel on a single GPU.\n\n'
            "    Exposes the wrapped model as .module (matching DDP's interface, since\n"
            "    checkpoint save/load and mid-training flag access assume DDP wrapping)\n"
            "    while forwarding calls straight through with zero synchronization --\n"
            "    and, critically, without touching torch.distributed at all.\n"
            '    """\n\n'
            "    def __init__(self, module):\n"
            "        super().__init__()\n"
            "        self.module = module\n\n"
            "    def forward(self, *args, **kwargs):\n"
            "        return self.module(*args, **kwargs)\n"
        )
        anchor = "from text.symbols import symbols"
        if anchor not in text:
            print("Warning: could not find symbols-import anchor; skipping DDP bypass patch.")
            return
        text = text.replace(anchor, anchor + wrapper_class, 1)

    import re

    ddp_pattern = re.compile(
        r"DDP\(\s*(net_g|net_d|net_dur_disc)\s*,\s*device_ids=\[rank\]\s*,\s*"
        r"find_unused_parameters=(?:True|False)\s*,?\s*\)",
        re.DOTALL,
    )
    text, count = ddp_pattern.subn(r"_SingleProcessModelWrapper(\1)", text)

    if text != original:
        train_py.write_text(text, encoding="utf-8")
        print(
            f"Patched {train_py.name}: DistributedDataParallel -> _SingleProcessModelWrapper "
            f"({count} occurrence(s)); torch.distributed is no longer used for gradient sync."
        )


def patch_cudnn_benchmark():
    """Disable cuDNN benchmark to prevent spiky GPU usage with variable length inputs.

    With variable length audio sequences, cuDNN benchmark forces the GPU to
    re-benchmark convolution algorithms for almost every batch, causing massive
    stalls and very spiky, low overall GPU utilization.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    original = train_py.read_text(encoding="utf-8")
    patched = original.replace("torch.backends.cudnn.benchmark = True", "torch.backends.cudnn.benchmark = False")
    if patched != original:
        print(f"Patched {train_py.name}: torch.backends.cudnn.benchmark -> False (prevent spiky GPU usage).")
        train_py.write_text(patched, encoding="utf-8")


def verify_windows_patches():
    """Fail loudly if any Windows compatibility patch didn't actually stick.

    Every patch function above is a best-effort text rewrite of a third-party
    file we don't control the exact formatting of. If an anchor pattern ever
    fails to match (a different backend version, a file locked by an editor,
    a permissions issue, etc.), the patch function silently no-ops rather
    than raising -- which would leave train.py in its old, crashing state
    while everything else proceeds as if it were fixed. Re-read the file and
    assert every patch is actually present before launching, so a mismatch
    surfaces immediately instead of producing another confusing identical
    crash.
    """
    train_py = BACKEND_DIR / "train.py"
    if not train_py.is_file():
        return
    text = train_py.read_text(encoding="utf-8")
    problems = []
    if "num_workers=8" in text:
        problems.append("DataLoader num_workers patch did not apply (still finds 'num_workers=8').")
    if 'backend="nccl"' in text:
        problems.append("Distributed backend patch did not apply (still finds 'backend=\"nccl\"').")
    if "_SingleProcessModelWrapper" not in text:
        problems.append("DDP-bypass class was not inserted into train.py.")
    if re.search(r"DDP\(\s*net_g\s*,", text):
        problems.append("DDP-bypass patch did not apply (still finds 'DDP(net_g, ...)').")
    if "torch.backends.cudnn.benchmark = True" in text:
        problems.append("cuDNN benchmark patch did not apply (still finds 'benchmark = True').")
    if problems:
        raise RuntimeError(
            "Windows compatibility patches did not fully apply to "
            f"{train_py}:\n  - " + "\n  - ".join(problems) + "\n"
            "This usually means the file is open/locked in another program (e.g. an "
            "editor), a permissions issue, or a different vits2_pytorch version than "
            "expected. Close any programs that might have train.py open, confirm "
            f"{BACKEND_DIR} is the repo you expect, and re-run."
        )
    print("Verified: all Windows compatibility patches are present in train.py.")


def require_backend():
    required = [BACKEND_DIR / "train.py", BACKEND_DIR / "models.py", BACKEND_DIR / "monotonic_align"]
    if all(path.exists() for path in required):
        return
    raise RuntimeError(
        "VITS2 backend is not installed. Run:\n"
        f"  {VENV_PYTHON.name} train_vits2.py --install-backend"
    )


def read_metadata(metadata_file, dataset_dir):
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    records = []
    skipped = 0
    with metadata_file.open("r", encoding="utf-8") as metadata:
        for line_number, line in enumerate(metadata, 1):
            parts = line.rstrip("\n").split("|", 1)
            if len(parts) != 2 or not parts[1].strip():
                skipped += 1
                continue
            audio_file = (dataset_dir / parts[0]).resolve()
            if not audio_file.is_file():
                print(f"Warning: missing WAV at metadata line {line_number}: {audio_file}")
                skipped += 1
                continue
            records.append((audio_file, parts[1].strip()))
    if not records:
        raise ValueError("No valid WAV/text records found in metadata.")
    if skipped:
        print(f"Skipped {skipped} malformed, empty, or missing-audio metadata rows.")
    return records


def validate_symbols(records):
    """Fail early if ZIPA emitted IPA symbols absent from the VITS2 tokenizer."""
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        from text.symbols import symbols
    except ImportError as exc:
        raise RuntimeError("Could not import VITS2 text symbols; run --install-backend first.") from exc
    supported = set(symbols)
    unsupported = sorted({character for _, text in records for character in text if character not in supported})
    if unsupported:
        rendered = " ".join(repr(character) for character in unsupported)
        raise ValueError(
            "The selected VITS2 backend does not define these metadata symbols: "
            f"{rendered}. Add them to {BACKEND_DIR / 'text' / 'symbols.py'} before training."
        )


def write_filelists(records, val_fraction):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("At least two records are required for a train/validation split.")
    shuffled = list(records)
    random.Random(42).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * val_fraction))
    validation_records = shuffled[:validation_count]
    training_records = shuffled[validation_count:]
    filelists_dir = PROJECT_DIR / "vits2_tamil_filelists"
    filelists_dir.mkdir(exist_ok=True)

    def write_list(path, rows):
        with path.open("w", encoding="utf-8", newline="\n") as filelist:
            for audio_file, text in rows:
                filelist.write(f"{audio_file}|{text}\n")

    train_path = filelists_dir / "train.txt"
    validation_path = filelists_dir / "validation.txt"
    write_list(train_path, training_records)
    write_list(validation_path, validation_records)
    return train_path, validation_path, len(training_records), len(validation_records)


def write_config(args, train_path, validation_path):
    """Create a conservative single-speaker, FP16 VITS2 configuration."""
    config = {
        "train": {
            "log_interval": 50,
            "eval_interval": args.eval_interval,
            "seed": 42,
            "epochs": args.epochs,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": args.batch_size,
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "use_mel_posterior_encoder": True,
            "training_files": str(train_path),
            "validation_files": str(validation_path),
            "text_cleaners": [],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": False,
        },
        "model": {
            "use_mel_posterior_encoder": True,
            "use_transformer_flows": True,
            "transformer_flow_type": "pre_conv",
            "use_spk_conditioned_encoder": False,
            "use_noise_scaled_mas": True,
            "use_duration_discriminator": True,
            # train.py reads hps.model.duration_discriminator_type directly
            # (no default) whenever use_duration_discriminator is True, so it
            # must be present here. "dur_disc_1" is VITS2's original duration
            # discriminator; use "dur_disc_2" for the alternate architecture.
            "duration_discriminator_type": "dur_disc_1",
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "use_sdp": False,
        },
    }
    config_path = PROJECT_DIR / "vits2_tamil_config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
    return config_path


def main():
    args = parse_args()
    if args.install_backend:
        install_backend()
        print("VITS2 backend installation complete.")
        return 0
    require_backend()
    patch_dataloader_workers(args.dataloader_workers)
    patch_windows_distributed_backend()
    patch_find_unused_parameters()
    patch_skip_ddp_single_gpu()
    patch_cudnn_benchmark()
    verify_windows_patches()
    dataset_dir = Path(args.dataset_dir)
    records = read_metadata(dataset_dir / args.metadata, dataset_dir)
    validate_symbols(records)
    train_path, validation_path, train_count, validation_count = write_filelists(records, args.val_fraction)
    config_path = write_config(args, train_path, validation_path)
    print(f"Prepared VITS2 data: {train_count} training / {validation_count} validation samples.")
    print(f"Launching VITS2 from {BACKEND_DIR}...")
    launch_env = os.environ.copy()
    if not args.disable_crash_diagnostics:
        # Dumps a real Python stack trace on fatal native crashes (e.g. Windows
        # STATUS_ACCESS_VIOLATION) instead of just an opaque process exit code.
        launch_env["PYTHONFAULTHANDLER"] = "1"
        # Makes CUDA kernel errors surface synchronously, with a message pointing
        # at the offending op, instead of silently corrupting the CUDA context
        # and crashing later somewhere unrelated-looking.
        launch_env["CUDA_LAUNCH_BLOCKING"] = "1"
        print("Crash diagnostics enabled (PYTHONFAULTHANDLER=1, CUDA_LAUNCH_BLOCKING=1).")
    result = subprocess.run(
        [sys.executable, "train.py", "-c", str(config_path), "-m", MODEL_NAME],
        cwd=BACKEND_DIR,
        env=launch_env,
    )
    return result.returncode


if __name__ == "__main__":
    restart_in_project_venv()
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"VITS2 setup error: {exc}", file=sys.stderr)
        raise SystemExit(1)