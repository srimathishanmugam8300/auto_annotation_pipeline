
"""Unified Audio Annotation Pipeline
Option 1: Transcription (Audio → Text)
Option 2: Sound Events Detection (Audio → Environmental Sounds)
Option 3: Transcription + Diarization (Who said what)
"""
import sys
import os
from pathlib import Path
import whisper
from datetime import datetime
import json
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import torch
from pyannote.audio import Pipeline

def transcribe_audio(input_path, output_folder):
    """Convert audio to text using Whisper"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Transcribe
    print(f"Transcribing: {input_path}")
    result = model.transcribe(input_path)
    
    # Extract text
    text = result["text"].strip()
    
    # Save to file
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"{input_name}_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\n✓ Transcription saved: {output_file}")
    print(f"\nTranscription:\n{text}")
    
    return output_file

def transcribe_with_diarization(input_path, output_folder):
    """Transcribe audio with speaker diarization (who said what)"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print("🎤 Starting Transcription + Diarization...")
    print("="*60)
    
    # Load Whisper model
    print("\n[1/3] Loading Whisper transcription model...")
    whisper_model = whisper.load_model("base")
    print("✓ Whisper model loaded")

    # Load diarization pipeline
    print("\n[2/3] Loading speaker diarization model...")
    print("(This may take a moment on first run)")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="hf_YOUR_TOKEN_HERE"  # Replace with your Hugging Face token
        )
        # Use GPU if available
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
        print("✓ Diarization model loaded")
    except Exception as e:
        print(f"\n⚠️  Diarization model loading failed: {e}")
        print("\nTo use this feature, you need a Hugging Face token:")
        print("1. Create account at https://huggingface.co")
        print("2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Get token from https://huggingface.co/settings/tokens")
        print("4. Update the code with your token\n")
        return None
    
    # Perform diarization
    print(f"\n[3/3] Analyzing speakers in: {input_path}")
    diarization = diarization_pipeline(input_path)
    
    # Load audio for transcription
    audio = whisper.load_audio(input_path)
    
    # Process each speaker segment
    print("\n" + "-"*60)
    print("Transcribing speaker segments...")
    print("-"*60 + "\n")
    
    speakers_data = []
    segment_count = 0
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_count += 1
        start_time = turn.start
        end_time = turn.end
        
        # Extract audio segment
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        audio_segment = audio[start_sample:end_sample]
        
        # Transcribe segment
        try:
            result = whisper_model.transcribe(audio_segment)
            text = result["text"].strip()
            
            if text:  # Only add if there's actual speech
                speakers_data.append({
                    "speaker": f"Speaker {speaker}",
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "duration": round(end_time - start_time, 2),
                    "text": text
                })
                
                print(f"[{start_time:06.2f}s - {end_time:06.2f}s] Speaker {speaker}:")
                print(f"  {text}")
                print()
        except Exception as e:
            print(f"⚠️  Error transcribing segment {segment_count}: {e}")
    
    # Create output data
    output_data = {
        "input_file": os.path.basename(input_path),
        "total_speakers": len(set(s["speaker"] for s in speakers_data)),
        "total_segments": len(speakers_data),
        "speakers": speakers_data
    }
    
    # Save JSON file
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_folder, f"{input_name}_diarization_{timestamp}.json")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save readable text file
    txt_file = os.path.join(output_folder, f"{input_name}_diarization_{timestamp}.txt")
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("SPEAKER-LABELED TRANSCRIPT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File: {os.path.basename(input_path)}\n")
        f.write(f"Total Speakers: {output_data['total_speakers']}\n")
        f.write(f"Total Segments: {output_data['total_segments']}\n\n")
        f.write("=" * 60 + "\n\n")
        
        for segment in speakers_data:
            timestamp_str = f"[{segment['start']:06.2f} - {segment['end']:06.2f}]"
            f.write(f"{timestamp_str} {segment['speaker']}:\n")
            f.write(f"{segment['text']}\n\n")
    
    # Summary
    print("="*60)
    print("✅ TRANSCRIPTION + DIARIZATION COMPLETE")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"  • Total speakers detected: {output_data['total_speakers']}")
    print(f"  • Total segments: {output_data['total_segments']}")
    print(f"\n💾 Output files:")
    print(f"  • JSON: {json_file}")
    print(f"  • Text: {txt_file}")
    print()
    
    return json_file

def detect_sound_events(input_path, output_folder, model=None):
    """Detect environmental sound events using YAMNet with robust post-processing"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YAMNet model from TensorFlow Hub (reuse if provided)
    if model is None:
        print("Loading YAMNet model for sound event detection...")
        model = hub.load('https://tfhub.dev/google/yamnet/1')
    else:
        print("Using cached model...")
    
    # Load audio file
    print(f"Analyzing audio: {input_path}")
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # ✅ FIX 1: Calculate real audio duration
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # Run model
    scores, embeddings, spectrogram = model(audio)
    
    # Get class names
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        for line in f:
            class_names.append(line.strip().split(',')[2])  # Get display name
    
    # Process detections
    sound_events = []
    frame_duration = 0.96  # YAMNet frame duration in seconds
    
    for i, frame_scores in enumerate(scores.numpy()):
        # Get top prediction for this frame
        top_class_idx = np.argmax(frame_scores)
        confidence = float(frame_scores[top_class_idx])
        
        # ✅ FIX 2: Stronger confidence filtering (0.8 instead of 0.7)
        if confidence >= 0.8:
            start_time = i * frame_duration
            end_time = min(start_time + frame_duration, audio_duration)  # ✅ FIX 1: Clip to real duration
            
            # Skip if beyond audio duration
            if start_time >= audio_duration:
                continue
            
            # ✅ FIX 3: Merge repeated labels (improved merging logic)
            if (sound_events and 
                sound_events[-1]['label'] == class_names[top_class_idx] and
                abs(sound_events[-1]['end'] - start_time) < 0.5):  # More aggressive merging
                # Extend previous event
                sound_events[-1]['end'] = round(end_time, 2)
                sound_events[-1]['confidence'] = max(sound_events[-1]['confidence'], confidence)
            else:
                # Add new event
                sound_events.append({
                    'label': class_names[top_class_idx],
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'confidence': round(confidence, 2)
                })
    
    # ✅ FIX 4: Dominant sound logic - find most frequent sound
    if sound_events:
        # Count total duration per label
        label_durations = {}
        for event in sound_events:
            duration = event['end'] - event['start']
            label = event['label']
            label_durations[label] = label_durations.get(label, 0) + duration
        
        # Find dominant sound (appears >50% of detected time)
        total_detected_time = sum(label_durations.values())
        dominant_sounds = []
        for label, duration in sorted(label_durations.items(), key=lambda x: x[1], reverse=True):
            ratio = duration / total_detected_time
            if ratio >= 0.5 or len(dominant_sounds) == 0:  # Keep dominant or at least one
                dominant_sounds.append({
                    'label': label,
                    'total_duration': round(duration, 2),
                    'coverage': round(ratio * 100, 1)
                })
            elif ratio >= 0.15:  # Keep significant secondary sounds
                dominant_sounds.append({
                    'label': label,
                    'total_duration': round(duration, 2),
                    'coverage': round(ratio * 100, 1)
                })
        
        # Filter events: keep only dominant/significant sounds
        significant_labels = {s['label'] for s in dominant_sounds}
        sound_events = [e for e in sound_events if e['label'] in significant_labels]
    
    # Create output JSON
    output_data = {
        'input_file': os.path.basename(input_path),
        'audio_duration': round(audio_duration, 2),
        'dominant_sounds': dominant_sounds if sound_events else [],
        'sound_events': sound_events
    }
    
    # Save to JSON file
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"{input_name}_events_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Sound events saved: {output_file}")
    
    # Enhanced output display
    if sound_events:
        print(f"\n🎯 Dominant Sounds:")
        for sound in dominant_sounds:
            print(f"  • {sound['label']}: {sound['total_duration']}s ({sound['coverage']}% coverage)")
        
        print(f"\n📋 Detected {len(sound_events)} sound events:")
        for event in sound_events[:10]:  # Show first 10
            print(f"  {event['start']:.2f}s - {event['end']:.2f}s: {event['label']} ({event['confidence']:.2f})")
        if len(sound_events) > 10:
            print(f"  ... and {len(sound_events) - 10} more events")
    else:
        print("\n⚠️  No high-confidence sound events detected (try different audio or lower threshold)")
    
    return output_file, output_data, model

def process_folder_with_sorting(input_folder, output_folder):
    """Process all audio files in a folder and sort by dominant sound"""
    
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    if not audio_files:
        print(f"⚠️  No audio files found in {input_folder}")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio files to process\n")
    print("="*60)
    
    # Load model once for all files
    print("Loading YAMNet model for sound event detection...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("✓ Model loaded\n")
    
    # Process each file
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        print("-" * 60)
        
        try:
            # Detect events (use temporary output folder)
            temp_output = output_folder
            output_file, output_data, model = detect_sound_events(
                str(audio_file), 
                temp_output,
                model=model
            )
            
            # Get dominant sound for sorting
            if output_data['dominant_sounds']:
                dominant_label = output_data['dominant_sounds'][0]['label']
            else:
                dominant_label = 'Other'
            
            results.append({
                'file': audio_file.name,
                'dominant': dominant_label,
                'output_data': output_data,
                'temp_output': output_file
            })
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            results.append({
                'file': audio_file.name,
                'dominant': 'Error',
                'output_data': None,
                'temp_output': None
            })
    
    # Sort files into category folders
    print("\n" + "="*60)
    print("📂 Sorting files by dominant sound...")
    print("="*60 + "\n")
    
    sorted_count = {}
    for result in results:
        if result['output_data'] is None:
            continue
            
        category = result['dominant']
        
        # Create category folder
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        
        # Save JSON in category folder
        input_name = Path(result['file']).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(category_folder, f"{input_name}_events_{timestamp}.json")
        
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(result['output_data'], f, indent=2)
        
        # Remove temp file if it exists
        if result['temp_output'] and os.path.exists(result['temp_output']):
            try:
                os.remove(result['temp_output'])
            except:
                pass
        
        # Track counts
        sorted_count[category] = sorted_count.get(category, 0) + 1
        print(f"  ✓ {result['file']} → {category}/")
    
    # Summary
    print("\n" + "="*60)
    print("✅ SORTING COMPLETE")
    print("="*60)
    print(f"\n📊 Distribution:")
    for category, count in sorted(sorted_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category}: {count} file(s)")
    print(f"\n📁 Output location: {output_folder}")
    print()

def main():
    # Check for --sort flag
    has_sort_flag = '--sort' in sys.argv
    
    # Remove --sort from args for processing
    args = [arg for arg in sys.argv if arg != '--sort']
    
    if len(args) < 4:
        print("Usage: python auto_pipeline.py <input> <output_folder> <option> [--sort]")
        print("")
        print("Options:")
        print("  1 - Transcription (speech to text)")
        print("  2 - Sound Events Detection (environmental sounds)")
        print("  3 - Transcription + Diarization (who said what)")
        print("")
        print("Flags:")
        print("  --sort - Process folder and sort by dominant sound (only with option 2)")
        print("")
        print("Examples:")
        print("  python auto_pipeline.py input/audio.wav output 1")
        print("  python auto_pipeline.py input/audio.wav output 2")
        print("  python auto_pipeline.py input/mixed_sounds output 2 --sort")
        print("  python auto_pipeline.py input/meeting.wav output 3")
        sys.exit(1)
    
    input_path = args[1]
    output_folder = args[2]
    option = args[3]
    
    # Validate input
    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
    
    # Check if input is folder or file
    is_folder = os.path.isdir(input_path)
    
    if option == "1":
        if has_sort_flag:
            print("⚠️  --sort flag only works with option 2 (Sound Events Detection)")
            sys.exit(1)
        if is_folder:
            print("⚠️  Folder processing not supported for transcription (option 1)")
            sys.exit(1)
        # Run transcription
        transcribe_audio(input_path, output_folder)
    elif option == "2":
        if has_sort_flag:
            if not is_folder:
                print("⚠️  --sort flag requires a folder input, not a single file")
                sys.exit(1)
            # Run folder sorting
            process_folder_with_sorting(input_path, output_folder)
        else:
            if is_folder:
                print("⚠️  Folder detected. Did you mean to use --sort flag?")
                print("      python auto_pipeline.py", input_path, output_folder, "2 --sort")
                sys.exit(1)
            # Run sound events detection on single file
            output_file, _, _ = detect_sound_events(input_path, output_folder)
    elif option == "3":
        if has_sort_flag:
            print("⚠️  --sort flag not supported for option 3 (Diarization)")
            sys.exit(1)
        if is_folder:
            print("⚠️  Folder processing not supported for diarization (option 3)")
            sys.exit(1)
        # Run transcription with diarization
        transcribe_with_diarization(input_path, output_folder)
    else:
        print(f"Error: Invalid option '{option}'. Use 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()
