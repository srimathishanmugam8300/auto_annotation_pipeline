# 🎙️ Speech-to-Text Transcription

Simple audio transcription pipeline.

## 🎧 Audio → 📝 Text

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python auto_pipeline.py <input_audio> <output_folder> 1
```

## Example

```bash
python auto_pipeline.py input/audio.wav output 1
```

## Output

- Plain text file with transcription
- Format: `{filename}_{timestamp}.txt`
