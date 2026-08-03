import os
import sys
import subprocess
import torch
from scipy.io.wavfile import write
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "veenv" / "Scripts" / "python.exe"
BACKEND_DIR = PROJECT_DIR / "vits2_pytorch"
CONFIG_PATH = PROJECT_DIR / "vits2_tamil_config.json"
CHECKPOINT_DIR = BACKEND_DIR / "logs" / "tamil_vits2"
OUT_DIR = PROJECT_DIR / "vits_output"

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

# Early restart so we are in the correct python environment
restart_in_project_venv()

# Add backend dir to sys.path so we can import its modules
sys.path.insert(0, str(BACKEND_DIR))

try:
    import commons
    import utils
    from models import SynthesizerTrn
    from text.symbols import symbols
    from text import text_to_sequence
except ImportError as e:
    print(f"❌ Error importing VITS2 backend modules: {e}")
    print("Please ensure you have run 'python train_vits2.py --install-backend' first.")
    sys.exit(1)

def find_latest_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None
    files = [f for f in CHECKPOINT_DIR.iterdir() if f.name.startswith("G_") and f.name.endswith(".pth")]
    if not files:
        return None
    # Sort files by modification time
    files.sort(key=lambda x: x.stat().st_mtime)
    return str(files[-1])

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main():
    print("Testing Trained Tamil VITS2 Model...")
    
    latest_ckpt = find_latest_checkpoint()
    
    manual_ckpt = input(f"Enter generator checkpoint path (Press Enter to use auto-detected: {latest_ckpt}): ").strip()
    if manual_ckpt:
        latest_ckpt = manual_ckpt
        
    if not latest_ckpt or not os.path.exists(latest_ckpt):
        print("❌ Error: Valid checkpoint file not found. Please provide a valid path.")
        return
        
    if not CONFIG_PATH.exists():
        print(f"❌ Error: Config file not found at {CONFIG_PATH}")
        return

    print(f"\n======================================")
    print(f"🚀 Loading Checkpoint: {latest_ckpt}")
    print(f"⚙️  Loading Config:     {CONFIG_PATH}")
    print(f"======================================\n")

    hps = utils.get_hparams_from_file(str(CONFIG_PATH))

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(latest_ckpt, net_g, None)
    print(f"\n✅ Model loaded successfully on {'GPU (CUDA)' if use_cuda else 'CPU'}!\n")

    # Initialize Epitran for Tamil
    try:
        import io
        import pandas as pd
        import pathlib
        
        # Windows console fix for printing IPA phonemes
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        
        # Patch for older pandas versions where .map() doesn't exist on DataFrame (panphon uses it)
        if not hasattr(pd.DataFrame, 'map') and hasattr(pd.DataFrame, 'applymap'):
            pd.DataFrame.map = pd.DataFrame.applymap
            
        # Patch pathlib.Path.open to force utf-8 so panphon can read its mapping files correctly
        _orig_path_open = pathlib.Path.open
        def _utf8_path_open(self, *args, **kwargs):
            mode = kwargs.get('mode', args[0] if args else 'r')
            if 'encoding' not in kwargs and 'b' not in mode:
                kwargs['encoding'] = 'utf-8'
            return _orig_path_open(self, *args, **kwargs)
        pathlib.Path.open = _utf8_path_open

        import epitran
        print("\n⏳ Loading Tamil G2P (Epitran)...")
        epi = epitran.Epitran('tam-Taml')
        
        pathlib.Path.open = _orig_path_open
    except ImportError:
        print("❌ Error: 'epitran' is not installed. Please run: pip install epitran")
        return
    except Exception as e:
        if '_orig_path_open' in locals():
            pathlib.Path.open = _orig_path_open
        print(f"❌ Error initializing epitran: {e}")
        return

    OUT_DIR.mkdir(exist_ok=True)

    default_text = "வணக்கம்! நான் ஒரு ஏய் குரல் மாதிரி பேசுகிறேன். உங்களை சந்திப்பதில் மிக்க மகிழ்ச்சி."
    print("\nType your Tamil text to synthesize, or type 'exit' to quit.")
    
    counter = 1
    while True:
        text = input(f"\n[Test {counter}] Text (Press Enter for default): ").strip()
        
        if text.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
            
        if not text:
            text = default_text
            
        output_wav = OUT_DIR / f"test_vits2_output_{counter}.wav"
        
        try:
            # Convert Tamil text to IPA phonemes
            phonemes = epi.transliterate(text)
            
            print(f"✍️  Original Text: {text}")
            print(f"🔤 Phonemes (IPA): {phonemes}")
            print(f"🎧 Synthesizing...")
            
            stn_tst = get_text(phonemes, hps)
            with torch.no_grad():
                x_tst = stn_tst.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                audio = (
                    net_g.infer(
                        x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )

            write(data=audio, rate=hps.data.sampling_rate, filename=str(output_wav))
            print(f"✅ Saved audio to: {output_wav.resolve()}")
            counter += 1
        except Exception as e:
            print(f"❌ Error during synthesis: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
