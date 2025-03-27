# 🎙️ CodeMixed-BanglaEnglish-ASR
🚀 **Automatic Speech Recognition for Code-Mixed Bangla-English Language**

This repository contains an **ASR model** designed for recognizing Bangla and English speech in a code-mixed setting. The model is trained using Transformer-based architectures and fine-tuned on a custom dataset.

## 📌 **Features**
✅ Supports **Bangla-English** code-mixed speech  
✅ Trained with **Wave2Vec2** architecture  
✅ Outputs transcriptions in **text format**  
✅ Includes pre-trained tokenizer and model weights  

---

## 🔧 **Installation**
Ensure you have Python **3.8+** installed. Run the following to install dependencies:

```sh
pip install datasets evaluate transformers[sentencepiece] librosa jiwer bangla-python collection openpyxl
```

## 📂 Dataset Details

This ASR model is trained on a custom code-mixed Bangla-English speech dataset. It contains labeled audio-text pairs collected from various sources, ensuring diverse linguistic coverage.

📌 **Dataset Highlights:**
✅ Includes both Bangla and English speech segments
✅ Preprocessed to match ASR model requirements
✅ Available in train/ and test/ splits

🔍 For more details on dataset format, structure, and preprocessing steps, check the **data.md** file.

## 🛠️ How to Use This Repository
**Clone this repository:**
```bash
git clone https://github.com/fayez94/CodeMixed-BanglaEnglish-ASR.git
cd CodeMixed-BanglaEnglish-ASR
```

## 📥 Loading the Model
**Load the model and tokenizer from the repository:**
```bash
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Load processor and model
model_path = "./"  # Use local directory if cloned
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

print("Model and processor loaded successfully!")
```

## 🎤 Running Inference
**Use the model to transcribe an audio file:**
```bash
import torch
import librosa

# Load and preprocess audio
audio_path = "path/to/audio.wav"  # Replace with actual audio file
speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

# Tokenize input
input_values = processor(speech_array, return_tensors="pt", padding=True).input_values

# Run model inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode output
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("Transcription:", transcription)
```

## 🏋️ Training (Optional)
**If you want to fine-tune the model with new data, prepare a dataset and follow the steps:**
```bash
python train.py --dataset path/to/dataset --epochs 5   #replace path/to/dataset with your actual dataset path
```

## 🗂 Repository Structure
```
CodeMixed-BanglaEnglish-ASR/
│── runs/                       # Training logs
│── model.safetensors           # Pre-trained model weights
│── config.json                 # Model configuration
│── preprocessor_config.json    # Processing configurations
│── tokenizer_config.json       # Tokenizer settings
│── vocab.json                  # Vocabulary file
│── merges.txt                  # BPE merges
│── training_args.bin           # Training arguments
│── notebook.md                 # notebook for training and inference
│── data.md                     # text and audio data for both train and test
│── README.md                   # Documentation
```







