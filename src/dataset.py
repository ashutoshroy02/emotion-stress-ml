"""Dataset classes for voice stress detection."""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


# Global label mapping for stress and emotion classes
LABEL_MAP = {
    "calm": 0,
    "angry": 1,
    "fearful": 2,
    "nervous": 3,
    "high_stress": 4,
    # Add more labels as needed for your dataset
    "neutral": 5,
    "happy": 6,
    "sad": 7,
    "disgust": 8,
    "surprised": 9,
}

# Inverse mapping for getting label names from IDs
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class SpeechDataset(Dataset):
    """Dataset for speech emotion and stress detection."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        feature_type: str = "mfcc",
        sr: int = 16000,
        max_duration: float = 10.0,
        transform=None,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing audio files
            feature_type: Type of features to extract ('mfcc', 'mel', or 'raw')
            sr: Sample rate for audio loading
            max_duration: Maximum duration in seconds (longer audios will be truncated)
            transform: Optional transform to be applied to features
        """
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.sr = sr
        self.max_duration = max_duration
        self.transform = transform
        
        # TODO: Initialize dataset
        # 1. Scan data_dir for audio files
        # 2. Create list of (audio_path, label) pairs
        # 3. Initialize any necessary feature extractors
        
        # Placeholder implementation
        self.samples = []  # List of (audio_path, label) tuples
        self.classes = []  # List of class names
        self.class_to_idx = {}  # Dict mapping class name to index
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'features': Extracted features (torch.Tensor)
                - 'label': Emotion/stress label (int)
                - 'audio_path': Path to the audio file (str)
        """
        # TODO: Implement data loading and feature extraction
        # 1. Load audio file
        # 2. Extract features based on self.feature_type
        # 3. Apply transform if specified
        # 4. Return features and label
        
        # Placeholder implementation
        audio_path, label = "path/to/audio.wav", 0
        features = torch.randn(1, 64, 100)  # Dummy features
        
        if self.transform:
            features = self.transform(features)
            
        return {
            'features': features,
            'label': torch.tensor(label, dtype=torch.long),
            'audio_path': audio_path,
        }


def get_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Get data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        **dataset_kwargs: Additional arguments to pass to SpeechDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # TODO: Implement data loading and splitting
    # 1. Create full dataset
    # 2. Split into train/val/test
    # 3. Create data loaders
    
    # Placeholder implementation
    dataset = SpeechDataset(data_dir, **dataset_kwargs)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size
    
    # Split dataset
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def create_manifest_from_folder(
    audio_root: str,
    output_csv: str
) -> None:
    """Create a CSV manifest listing all audio files and their labels.
    
    Args:
        audio_root: Root directory containing subfolders named by labels
        output_csv: Path to save the CSV manifest file
    """
    audio_root_path = Path(audio_root)
    output_csv_path = Path(output_csv)
    
    # Input validation
    if not audio_root_path.exists():
        raise ValueError(f"Audio root directory does not exist: {audio_root}")
    
    if not audio_root_path.is_dir():
        raise ValueError(f"Audio root path is not a directory: {audio_root}")
    
    print(f"Creating manifest from {audio_root_path}...")
    print(f"Output will be saved to {output_csv_path}")
    
    # Collect all audio files and their labels
    manifest_data = []
    total_files = 0
    skipped_files = 0
    
    # Walk through all subdirectories
    for label_dir in audio_root_path.iterdir():
        if not label_dir.is_dir():
            continue
            
        label_name = label_dir.name.lower()
        
        # Check if label is in our mapping
        if label_name not in LABEL_MAP:
            print(f"Warning: Label '{label_name}' not found in LABEL_MAP. Skipping...")
            continue
        
        label_id = LABEL_MAP[label_name]
        print(f"Processing label: {label_name} (ID: {label_id})")
        
        # Find all .wav files in this label directory
        wav_files = list(label_dir.glob("*.wav"))
        
        if not wav_files:
            print(f"  No .wav files found in {label_dir}")
            continue
        
        print(f"  Found {len(wav_files)} .wav files")
        
        # Add each file to manifest
        for wav_file in wav_files:
            # Use relative path from audio_root for portability
            relative_path = wav_file.relative_to(audio_root_path)
            manifest_data.append({
                'path': str(relative_path),
                'label': label_name,
                'label_id': label_id
            })
            total_files += 1
    
    if not manifest_data:
        print("No valid audio files found!")
        return
    
    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write manifest to CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['path', 'label', 'label_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            for row in manifest_data:
                writer.writerow(row)
        
        print(f"\nManifest creation complete!")
        print(f"Total files processed: {total_files}")
        print(f"Files skipped: {skipped_files}")
        print(f"Manifest saved to: {output_csv_path}")
        
        # Print label distribution
        label_counts = {}
        for row in manifest_data:
            label = row['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            label_id = LABEL_MAP[label]
            print(f"  {label} (ID: {label_id}): {count} files")
            
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        raise


if __name__ == "__main__":
    # Example: Create manifest from preprocessed audio folder
    print("Creating audio manifest...")
    create_manifest_from_folder("../data/indian", "../data/indian_manifest.csv")
    
    # Example usage for data loaders
    print("\nTesting data loaders...")
    data_dir = "../data/global/RAVDESS"
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        feature_type="mfcc",
    )
    
    # Get a batch of data
    batch = next(iter(train_loader))
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch labels: {batch['label']}")
