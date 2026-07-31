# EL-TaVITS (Expressive & Lite Tamil VITS)  *****In Progress
(*title subject to change*)

This repository contains the training and inference code for **EL-TaVITS**, an experimental, low-resource Text-to-Speech (TTS) synthesis model for the Tamil language.

The primary focus of this project is **efficiency and low-resource training**. High-quality, expressive speech synthesis typically requires massive datasets and compute. This research explores achieving expressive Tamil TTS using a very limited amount of training data.

## Key Features

*   **Low-Resource Training**: The model in this repository is designed to be trained effectively on extremely limited datasets. The current experiments are based on **only ~16 hours of Tamil audio data**.
*   **Expressive Synthesis**: Despite the minimal data, the VITS architecture is tuned to produce natural, highly expressive phonetic outputs.
*   **Lightweight**: Focused on efficient architecture configurations suitable for resource-constrained environments.

## Repository Structure

*   `train_vits.py`: The main script used to train the VITS model on the prepared Tamil dataset.
*   `test_phonemes_vits.py`: Inference script to generate audio from Tamil text/phonemes using the trained model weights.
*   `requirements.txt`: Python dependencies required to run the training and inference scripts.

*(Note: Data extraction, normalization, and dataset preparation scripts are kept external to this repository to maintain a focused training and inference pipeline.)*

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd tamil-vits
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have a compatible PyTorch environment setup with CUDA if training on a GPU).*

## Dataset Preparation

To train the model, you must provide your audio data and transcriptions in a specific format compatible with Coqui TTS.

1. **Dataset Directory**: Create a folder named `tamil_vits_dataset` in the root of the repository.
2. **Audio Files**: Place all your `.wav` audio files inside this folder (or in subdirectories within it). 
   - **Sample Rate**: The model is configured for **22050 Hz**. Ensure your audio is resampled to this rate for optimal training.
   - **Format**: Mono channel, 16-bit PCM WAV is recommended.
3. **Metadata File**: Create a file named `metadata.csv` inside the `tamil_vits_dataset` folder.
   - The file must use the `id|text` format, where each line corresponds to an audio file and its Tamil transcript.
   - The `id` should be the relative path to the audio file from the dataset folder.
   - **Example `metadata.csv` format**:
     ```csv
     wavs/audio_001.wav|வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?
     wavs/audio_002.wav|இந்த இயந்திர கற்றல் மாதிரி நன்றாக வேலை செய்கிறது.
     ```

*Note: The training script will automatically convert this format into the `id|text|normalized_text` LJSpeech format required by Coqui TTS.*

## Usage

### Training
To begin training the model on your prepared dataset:
```bash
python train_vits.py
```
*Note: Make sure to adjust paths and hyperparameters within the script to match your local environment and dataset location.*

### Inference (Testing)
To generate speech using your trained checkpoint:
```bash
python test_phonemes_vits.py
```
This script allows you to synthesize audio by passing Tamil text/phonemes through the trained model. Modify the inference parameters (like `noise_scale`, `length_scale`) inside the script to adjust expressiveness and speed.

## Research Context
This work is conducted for research purposes, aiming to democratize speech synthesis for low-resource languages by reducing the barrier to entry regarding dataset size and computational power.
