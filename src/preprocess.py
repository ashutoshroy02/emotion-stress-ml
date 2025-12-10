"""Audio preprocessing utilities for voice stress detection."""
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def resample_and_normalize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000
) -> None:
    """Load, resample, normalize, and save an audio file.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio file
        target_sr: Target sample rate in Hz
    """
    try:
        # Load audio file
        print(f"Loading {input_path}...")
        audio, orig_sr = librosa.load(input_path, sr=None, mono=True)
        
        # Resample if necessary
        if orig_sr != target_sr:
            print(f"Resampling from {orig_sr}Hz to {target_sr}Hz...")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        
        # Normalize amplitude to [-1, 1] range
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                # Ensure we don't clip by scaling to 0.95 max amplitude
                audio = audio * 0.95
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        sf.write(output_path, audio, target_sr)
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")


def batch_preprocess_audio(
    input_dir: str,
    output_dir: str,
    target_sr: int = 16000,
    exts: tuple[str, ...] = (".wav", ".mp3")
) -> None:
    """Recursively preprocess all audio files in a directory.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed audio files
        target_sr: Target sample rate in Hz
        exts: Tuple of allowed file extensions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist!")
        return
    
    print(f"Starting batch preprocessing...")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Target sample rate: {target_sr}Hz")
    print(f"Supported extensions: {exts}")
    
    # Find all audio files recursively
    audio_files = []
    for ext in exts:
        audio_files.extend(input_path.rglob(f"*{ext}"))
        audio_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    processed = 0
    failed = 0
    
    for audio_file in audio_files:
        try:
            # Calculate relative path and output path
            rel_path = audio_file.relative_to(input_path)
            # Change extension to .wav for output
            output_file = output_path / rel_path.with_suffix('.wav')
            
            # Process the file
            resample_and_normalize_audio(
                str(audio_file),
                str(output_file),
                target_sr
            )
            processed += 1
            
        except Exception as e:
            print(f"Failed to process {audio_file}: {str(e)}")
            failed += 1
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed} files")
    print(f"Failed: {failed} files")


if __name__ == "__main__":
    # Example: preprocess indian_raw to indian
    batch_preprocess_audio("../data/indian_raw", "../data/indian", target_sr=16000)
