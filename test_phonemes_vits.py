import os
import torch
from TTS.utils.synthesizer import Synthesizer

# Directories matching train_vits.py
OUT_DIR = r"C:\vits_output"
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth") and "best" not in f]
    if not files:
        # fallback to any .pth file like _best
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
        if not files:
            return None
    
    # Sort files to find the latest
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
    return os.path.join(CHECKPOINT_DIR, files[-1])

def main():
    print("Testing Trained Tamil VITS Model...")
    
    # 1. Find checkpoint
    latest_ckpt = find_latest_checkpoint()
    
    # Allow user override
    manual_ckpt = input(f"Enter checkpoint path (Press Enter to use auto-detected: {latest_ckpt}): ").strip()
    if manual_ckpt:
        latest_ckpt = manual_ckpt
        
    if not latest_ckpt or not os.path.exists(latest_ckpt):
        print("❌ Error: Valid checkpoint file not found. Please provide a valid path.")
        return
        
    # 2. Find config
    config_path = os.path.join(os.path.dirname(latest_ckpt), "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Error: config.json not found in {os.path.dirname(latest_ckpt)}")
        # Check if it was saved locally in OUT_DIR
        alt_config = os.path.join(OUT_DIR, "config.json")
        if os.path.exists(alt_config):
            config_path = alt_config
            print(f"✅ Found config at {config_path}")
        else:
            manual_config = input("Enter config.json path: ").strip()
            if manual_config and os.path.exists(manual_config):
                config_path = manual_config
            else:
                return

    print(f"\n======================================")
    print(f"🚀 Loading Checkpoint: {latest_ckpt}")
    print(f"⚙️  Loading Config:     {config_path}")
    print(f"======================================\n")

    # 3. Initialize Synthesizer
    # Synthesizer wraps the model and does tokenization, AP logic automatically!
    try:
        use_cuda = torch.cuda.is_available()
        synth = Synthesizer(
            tts_checkpoint=latest_ckpt,
            tts_config_path=config_path,
            use_cuda=use_cuda
        )
        print(f"\n✅ Model loaded successfully on {'GPU (CUDA)' if use_cuda else 'CPU'}!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Initialize Epitran for Tamil
    try:
        import sys
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

    # 5. Interactive testing loop
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
            
        output_wav = f"vits_output\\test_phoneme_output_{counter}.wav"
        
        try:
            # Convert Tamil text to IPA phonemes
            phonemes = epi.transliterate(text)
            
            print(f"✍️  Original Text: {text}")
            print(f"🔤 Phonemes (IPA): {phonemes}")
            print(f"🎧 Synthesizing...")
            
            # Pass the phonemes to the model
            wav = synth.tts(
                phonemes,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0
            )
            synth.save_wav(wav, output_wav)
            print(f"✅ Saved audio to: {os.path.abspath(output_wav)}")
            counter += 1
        except Exception as e:
            print(f"❌ Error during synthesis: {e}")

if __name__ == "__main__":
    main()
