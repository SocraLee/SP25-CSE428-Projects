import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import polars as pl
from typing import List, Tuple
from tqdm import tqdm
from sklearn.decomposition import PCA  
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast
import os 
def preprocess_anesthesia_data(data_path, region_name=None):
    """
    Extract specific segments from each anesthesia phase and stitch them together.
    
    For each recording:
    - First 3 minutes of baseline
    - Middle 3 minutes of induction  
    - Middle 3 minutes of maintenance
    - Last 3 minutes of emergence
    
    - Total: 12 minutes per recording.
    
    Returns:
        df_final: Processed dataframe with 12-minute recordings
    """
    df_polars = pl.scan_parquet(data_path)

    if region_name is not 'all_regions':
        df_polars = df_polars.filter(pl.col("region_parent") == region_name)

    df_polars = (
        df_polars
        .select(["recording_name", "firing_rate_hz", "phase", "iso_conc%", "region_parent"])
        .collect()
    )

    df = df_polars.to_pandas()    
    
    processed_recordings = []
    
    print(f"Processing {len(df)} recordings for {region_name}...")
    
    for idx, row in df.iterrows():
    
        firing_rates = np.array(row['firing_rate_hz'], dtype=np.float32)
        phases = row['phase']
        
        # convert phases to list and clean
        phases = [str(p) if p is not None else 'unknown' for p in phases]
        
        # make sure arrays are same length
        min_length = min(len(firing_rates), len(phases))
        firing_rates = firing_rates[:min_length]
        phases = phases[:min_length]
        assert len(firing_rates) == len(phases)
        
        # Find phase transitions
        phase_segments = {}
        current_phase = phases[0]
        start_idx = 0
        
        for i in range(1, len(phases)):
            if phases[i] != current_phase:
                # Save previous phase segment
                if current_phase not in phase_segments:
                    phase_segments[current_phase] = []
                phase_segments[current_phase].append((start_idx, i-1))
                
                # Start new phase
                current_phase = phases[i]
                start_idx = i
        
        # last phase
        if current_phase not in phase_segments:
            phase_segments[current_phase] = []
        phase_segments[current_phase].append((start_idx, len(phases)-1))
        # phase_segments = {'baseline': [(0, 299)], 'induction': [(300, 959)], 'maintenance': [(960, 1559)], 'emergence': [(1560, 2759)]}

        # Extract required segments (3 minutes = 180 seconds each)
        extracted_firing_rates = []
        extracted_phases = []
        
        # 1. first 3 min of baseline (180 seconds)
        if 'baseline' in phase_segments and len(phase_segments['baseline']) > 0:
            baseline_start, baseline_end = phase_segments['baseline'][0]
            baseline_length = baseline_end - baseline_start + 1
            
            if baseline_length >= 180:
                # Take first 180 seconds
                segment_start = baseline_start
                segment_end = baseline_start + 180
                extracted_firing_rates.extend(firing_rates[segment_start:segment_end])
                extracted_phases.extend(['baseline'] * 180)
            else:
                print(f"Uh oh! Recording {idx} - baseline too short ({baseline_length}s), skipping")
                continue
        else:
            print(f"Uh oh! Recording {idx} - no baseline found, skipping")
            continue
        
        # 2. middle 3 min of induction
        if 'induction' in phase_segments and len(phase_segments['induction']) > 0:
            # Find longest induction segment (in case there are multiple)
            induction_start, induction_end = phase_segments['induction'][0]
            induction_length = induction_end - induction_start + 1
            
            if induction_length >= 180:
                # middle 180 seconds
                middle_start = induction_start + (induction_length - 180) // 2
                middle_end = middle_start + 180
                extracted_firing_rates.extend(firing_rates[middle_start:middle_end])
                extracted_phases.extend(['induction'] * 180)
            else:
                print(f"Uh oh! Recording {idx} - induction too short ({induction_length}s), skipping")
                continue
        else:
            print(f"Uh oh! Recording {idx} - no induction phase found, skipping")
            continue
        
        # 3. middle 3 minutes of maintenance
        if 'maintenance' in phase_segments and len(phase_segments['maintenance']) > 0:
            maintenance_start, maintenance_end = phase_segments['maintenance'][0]
            maintenance_length = maintenance_end - maintenance_start + 1
            
            if maintenance_length >= 180:
                # middle 180 seconds
                middle_start = maintenance_start + (maintenance_length - 180) // 2
                middle_end = middle_start + 180
                extracted_firing_rates.extend(firing_rates[middle_start:middle_end])
                extracted_phases.extend(['maintenance'] * 180)
            else:
                print(f"Uh oh! Recording {idx} - maintenance too short ({maintenance_length}s), skipping")
                continue
        else:
            print(f"Uh oh! Recording {idx} - no maintenance phase found, skipping")
            continue
        
        # 4. last 3 min of emergence
        if 'emergence' in phase_segments and len(phase_segments['emergence']) > 0:
            emergence_start, emergence_end = phase_segments['emergence'][0]
            emergence_length = emergence_end - emergence_start + 1
            
            if emergence_length >= 180:
                # last 180 seconds
                segment_start = emergence_end - 180 + 1
                segment_end = emergence_end + 1
                extracted_firing_rates.extend(firing_rates[segment_start:segment_end])
                extracted_phases.extend(['emergence'] * 180)
            else:
                print(f"Uh oh! Recording {idx} - emergence too short ({emergence_length}s), skipping")
                continue
        else:
            print(f"Uh oh! Recording {idx} - no emergence phase found, skipping")
            continue
        
        # Verify we have exactly 720 seconds (12 minutes)
        assert len(extracted_firing_rates) == 720 == len(extracted_phases)
        processed_recordings.append({
                'recording_name': row['recording_name'],
                'firing_rate_hz': extracted_firing_rates,
                'phase': extracted_phases,
                'region_parent': region_name,
                'original_index': idx
            })
                
    df_final = pd.DataFrame(processed_recordings)
    
    print(f"\n" + "="*60)
    print(f"PREPROCESSING COMPLETE")
    print(f"Original recordings: {len(df)}")
    print(f"Successfully processed: {len(df_final)}")
    print(f"="*60)
    
    return df_final

def collate_fn(batch):
    """
    to handle variable length sequences. Not needed once filtered to 3 min per phase
    
    """
    sequences, labels = zip(*batch)
    
    # get lengths
    lengths = [len(seq) for seq in sequences]
    max_length = max(lengths)
    
    # pad sequences to max length
    padded_sequences = torch.zeros(len(sequences), max_length, 1)
    padded_labels = torch.zeros(len(sequences), max_length, dtype=torch.long)
    
    # copy actual data
    for i, (seq, label) in enumerate(zip(sequences, labels)):
        length = len(seq)
        padded_sequences[i, :length] = seq
        padded_labels[i, :length] = label

    return padded_sequences, padded_labels, torch.tensor(lengths)

def visualize_attention(model, dataset, recording_idx=0, save_path=None, device = 'cpu'):
    """Visualize attention weights for a specific recording"""
    model.eval()
    
    # Get single rec
    sequence, labels = dataset[recording_idx]
    sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        outputs, attention_weights = model(sequence)
        predictions = torch.argmax(outputs, dim=-1)
    
    # to numpy
    attention_weights = attention_weights.cpu().numpy().squeeze()
    predictions = predictions.cpu().numpy().squeeze()
    true_labels = labels.numpy()
    sequence_data = sequence.cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    # plot firing rate
    axes[0].plot(sequence_data, 'b-', alpha=0.7)
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title(f'Recording {recording_idx}: Neural Activity and Attention Analysis')
    axes[0].grid(True, alpha=0.3)
    
    # plot attention weights
    axes[1].plot(attention_weights, 'r-', linewidth=2)
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Attention Weights Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # plot true vs predicted phases
    desired_order = ['baseline', 'induction', 'maintenance', 'emergence']
    phase_names = dataset.phase_names
    
    # tp map from og values to display order
    def map_to_display_order(encoded_labels):
        mapped_labels = np.zeros_like(encoded_labels)
        for i, encoded_val in enumerate(encoded_labels):
            phase_name = phase_names[encoded_val]
            if phase_name in desired_order:
                mapped_labels[i] = desired_order.index(phase_name)
            else:
                mapped_labels[i] = 0 
        return mapped_labels
    
    true_labels_mapped = map_to_display_order(true_labels)
    predictions_mapped = map_to_display_order(predictions)
    
    time_points = range(len(true_labels_mapped))
    axes[2].plot(time_points, true_labels_mapped, 'g-', label='True Phase', linewidth=2)
    axes[2].plot(time_points, predictions_mapped, 'orange', label='Predicted Phase', linewidth=2, alpha=0.7)
    axes[2].set_ylabel('Phase')
    axes[2].set_title('True vs Predicted Phases')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[2].set_yticks(range(len(desired_order)))
    axes[2].set_yticklabels(desired_order)
    axes[2].set_ylim(-0.5, len(desired_order) - 0.5)
    
    # attention as a heatmap too
    attention_2d = attention_weights.reshape(1, -1)
    im = axes[3].imshow(attention_2d, cmap='Reds', aspect='auto')
    axes[3].set_ylabel('Attention')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('Attention Heatmap')
    
    plt.colorbar(im, ax=axes[3])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'recording_{recording_idx}'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return attention_weights, predictions, true_labels