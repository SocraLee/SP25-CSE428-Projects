import os
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import pandas as pd

class PatientDicomDataset(Dataset):
    """Data set for regression, and has all gene values for y values"""
    def __init__(self, root_dir, csv_path, transform=None):
        #print("Starting dataset load")
        self.transform = transform
        self.patient_images = []
        self.patient_ids = []
        self.gene_values = []
        print("Loading CSV and images...")
        # Load CSV
        df = pd.read_csv(csv_path, dtype=str)
        df.set_index(df.columns[0], inplace=True)  # Set first column (genes) as index
        df = df.transpose()  # Now rows are patients, columns are genes
        df.index.name = 'patient_id'
        self.corresponding_genes = df.columns.tolist()
        
        # print("CSV loaded successfully.")
        # print("Number of patients in CSV:", len(df))
        # Collect images per patient
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
           
            if not os.path.isdir(patient_path):
                continue
            #add all images in the patient folder
            #gonna do away with metadata for positionality and such for now
            images = []
            for body_part_folder in os.listdir(patient_path):
                body_part_path = os.path.join(patient_path, body_part_folder)
                if not os.path.isdir(body_part_path):
                    continue
                for image_folder in os.listdir(body_part_path):
                    image_path = os.path.join(body_part_path, image_folder)
                    if not os.path.isdir(image_path):
                        continue

                    dcm_files = [f for f in os.listdir(image_path) if f.endswith('.dcm')]
                    dcm_files = sorted(dcm_files, key=extract_index_number)

                    # Keep only the 32 most central slices
                    if len(dcm_files) > 8:
                        center = len(dcm_files) // 2
                        dcm_files = dcm_files[center - 16 : center + 16]

                    for file in dcm_files:
                        full_path = os.path.join(image_path, file)
                        images.append(full_path)
            if patient_id in df.index:
                self.patient_images.append(images)
                self.patient_ids.append(patient_id)
                self.gene_values.append(df.loc[patient_id])
                #want to get the first row minus the first column

    def __len__(self):
        return len(self.patient_images)

    def __getitem__(self, idx):
        image_paths = self.patient_images[idx]
        patient_id = self.patient_ids[idx]
        raw_gene_values = self.gene_values[idx].values.astype(np.float32)
        raw_gene_values = np.nan_to_num(raw_gene_values, nan=0.0)
        gene_values = torch.tensor(raw_gene_values)
        assert not torch.isnan(gene_values).any(), "Still have NaNs after replacement"

        corresponding_genes = self.corresponding_genes

        images = []

        for path in image_paths:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)

                # Normalize image
                img -= img.min()
                if img.max() != 0:
                    img /= img.max()

                if img.ndim == 3:
                    # Volume: [D, H, W] → treat as multiple slices
                    for slice_img in img:
                        slice_tensor = torch.tensor(slice_img).unsqueeze(0)  # [1, H, W]
                        if self.transform:
                            slice_tensor = self.transform(slice_tensor)
                        images.append(slice_tensor)

                elif img.ndim == 2:
                    # Single slice: [H, W]
                    slice_tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
                    if self.transform:
                        slice_tensor = self.transform(slice_tensor)
                    images.append(slice_tensor)

                else:
                    print(f"⚠️ Unexpected shape {img.shape} in file {path} — skipping")

            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")

        return images, patient_id, gene_values, corresponding_genes

def collate_fn(batch):
    images, _, gene_values, _ = zip(*batch)

    processed_images = []
    max_depth = 0

    for img_list in images:
        # Each img: [1, H, W] → stack → [D, 1, H, W]
        stacked = torch.stack(img_list)  # [D, 1, H, W]
        max_depth = max(max_depth, stacked.shape[0])
        processed_images.append(stacked)

    # Pad to max depth
    padded_images = []
    for img in processed_images:
        d = img.shape[0]
        if d < max_depth:
            pad = torch.zeros((max_depth - d, *img.shape[1:]), device=img.device)
            img = torch.cat([img, pad], dim=0)
        padded_images.append(img)

    batch_images = torch.stack(padded_images)  # [B, D, 1, H, W]
    gene_tensor = torch.stack(gene_values)

    return batch_images, None, gene_tensor, None

def extract_index_number(filename):
    try:
        base = os.path.splitext(filename)[0]        # e.g., "AX-123"
        parts = base.split('-')
        number_part = parts[-1]                     # get "123"
        return int(number_part)
    except (IndexError, ValueError):
        return float('inf')  # Push malformed filenames to the end
    
class PatientDicomDatasetSingleGeneByIndex(Dataset):
    def __init__(self, root_dir, csv_path,index, transform=None):
        """takes an index and extracts the relevant gene from the csv file"""
        #print("Starting dataset load")
        self.transform = transform
        self.patient_images = []
        self.patient_ids = []
        self.gene_values = []
        print("Loading CSV and images...")
        # Load CSV
        df = pd.read_csv(csv_path, dtype=str)
        df.set_index(df.columns[0], inplace=True)  # Set first column (genes) as index
        df = df.transpose()  # Now rows are patients, columns are genes
        # now slice off the single gene
        df = df.iloc[:, index:index+1]  # Get only the specified gene column
        df.index.name = 'patient_id'
        # store the corresponding gene name as a string, not a list
        self.corresponding_gene = df.columns[0]
        
        # print("CSV loaded successfully.")
        # print("Number of patients in CSV:", len(df))
        # Collect images per patient
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
           
            if not os.path.isdir(patient_path):
                continue
            #add all images in the patient folder
            #gonna do away with metadata for positionality and such for now
            images = []
            for body_part_folder in os.listdir(patient_path):
                body_part_path = os.path.join(patient_path, body_part_folder)
                if not os.path.isdir(body_part_path):
                    continue
                for image_folder in os.listdir(body_part_path):
                    image_path = os.path.join(body_part_path, image_folder)
                    if not os.path.isdir(image_path):
                        continue

                    dcm_files = [f for f in os.listdir(image_path) if f.endswith('.dcm')]
                    dcm_files = sorted(dcm_files, key=extract_index_number)

                    # Keep only the 32 most central slices
                    if len(dcm_files) > 8:
                        center = len(dcm_files) // 2
                        dcm_files = dcm_files[center - 16 : center + 16]

                    for file in dcm_files:
                        full_path = os.path.join(image_path, file)
                        images.append(full_path)
            if patient_id in df.index:
                # print(images)
                gene_val= df.loc[patient_id].values[0]
                #if nan then skip over it and delete the images added to path
                if pd.isna(gene_val):
                    print(f"⚠️ Skipping patient {patient_id} due to NaN gene value for {self.corresponding_gene}")
                    #need to clear images of this patient
                    images.clear()
                    continue
                self.gene_values.append(gene_val)
                
                self.patient_images.append(images.copy())
                self.patient_ids.append(patient_id)
                images.clear()
                
                
                #want to get the first row minus the first column

    def __len__(self):
        return len(self.patient_images)

    def __getitem__(self, idx):
        image_paths = self.patient_images[idx]
        patient_id = self.patient_ids[idx]
        raw_gene_value = self.gene_values[idx]
        raw_gene_value = np.nan_to_num(raw_gene_value, nan=0.0)
        gene_values = raw_gene_value
        corresponding_genes = self.corresponding_gene

        images = []

        for path in image_paths:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)

                # Normalize image
                img -= img.min()
                if img.max() != 0:
                    img /= img.max()

                if img.ndim == 3:
                    # Volume: [D, H, W] → treat as multiple slices
                    for slice_img in img:
                        slice_tensor = torch.tensor(slice_img).unsqueeze(0)  # [1, H, W]
                        if self.transform:
                            slice_tensor = self.transform(slice_tensor)
                        images.append(slice_tensor)

                elif img.ndim == 2:
                    # Single slice: [H, W]
                    slice_tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
                    if self.transform:
                        slice_tensor = self.transform(slice_tensor)
                    images.append(slice_tensor)

                else:
                    print(f"⚠️ Unexpected shape {img.shape} in file {path} — skipping")

            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")

        return images, patient_id, gene_values, corresponding_genes

def collate_fn(batch):
    images, _, gene_values, _ = zip(*batch)

    processed_images = []
    max_depth = 0

    for img_list in images:
        # Each img: [1, H, W] → stack → [D, 1, H, W]
        stacked = torch.stack(img_list)  # [D, 1, H, W]
        max_depth = max(max_depth, stacked.shape[0])
        processed_images.append(stacked)

    # Pad to max depth
    padded_images = []
    for img in processed_images:
        d = img.shape[0]
        if d < max_depth:
            pad = torch.zeros((max_depth - d, *img.shape[1:]), device=img.device)
            img = torch.cat([img, pad], dim=0)
        padded_images.append(img)

    batch_images = torch.stack(padded_images)  # [B, D, 1, H, W]
    # Stack gene values if there are multiple genes
    #might only be one gene value and one gene per patient, if that is a the case things are stored in np.str and np.

    if isinstance(gene_values[0], torch.Tensor):
        gene_tensor = torch.stack(gene_values)  # [B, num_genes]
    elif isinstance(gene_values[0], (list, np.ndarray)):
        gene_tensor = torch.tensor(gene_values)  # [B, num_genes]
    else:
        # Assume single gene value per patient (e.g., float, int, or str convertible to float)
        gene_tensor = torch.tensor([float(g) for g in gene_values]).unsqueeze(1)  # [B, 1]

    return batch_images, None, gene_tensor, None

class PatientDicomDatasetSingleGeneByIndexClass(Dataset):

    def __init__(self, root_dir, csv_path,index, transform=None):
        """takes an index and extracts the relevant gene from the csv file"""
        #print("Starting dataset load")
        self.transform = transform
        self.patient_images = []
        self.patient_ids = []
        self.gene_values = []
        print("Loading CSV and images...")
        # Load CSV
        df = pd.read_csv(csv_path, dtype=str)
        df.set_index(df.columns[0], inplace=True)  # Set first column (genes) as index
        df = df.transpose()  # Now rows are patients, columns are genes
        # now slice off the single gene
        df = df.iloc[:, index:index+1]  # Get only the specified gene column
        df.index.name = 'patient_id'
        # store the corresponding gene name as a string, not a list
        self.corresponding_gene = df.columns[0]
        
        # print("CSV loaded successfully.")
        # print("Number of patients in CSV:", len(df))
        # Collect images per patient
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
           
            if not os.path.isdir(patient_path):
                continue
            #add all images in the patient folder
            #gonna do away with metadata for positionality and such for now
            images = []
            for body_part_folder in os.listdir(patient_path):
                body_part_path = os.path.join(patient_path, body_part_folder)
                if not os.path.isdir(body_part_path):
                    continue
                for image_folder in os.listdir(body_part_path):
                    image_path = os.path.join(body_part_path, image_folder)
                    if not os.path.isdir(image_path):
                        continue

                    dcm_files = [f for f in os.listdir(image_path) if f.endswith('.dcm')]
                    dcm_files = sorted(dcm_files, key=extract_index_number)

                    # Keep only the 32 most central slices
                    if len(dcm_files) > 8:
                        center = len(dcm_files) // 2
                        dcm_files = dcm_files[center - 16 : center + 16]

                    for file in dcm_files:
                        full_path = os.path.join(image_path, file)
                        images.append(full_path)
            if patient_id in df.index:
                # print(images)
                gene_val= df.loc[patient_id].values[0]
                #if nan then skip over it and delete the images added to path
                if pd.isna(gene_val):
                    print(f"⚠️ Skipping patient {patient_id} due to NaN gene value for {self.corresponding_gene}")
                    #need to clear images of this patient
                    images.clear()
                    continue
                self.gene_values.append(gene_val)
                
                self.patient_images.append(images.copy())
                self.patient_ids.append(patient_id)
                images.clear()
                
                
                #want to get the first row minus the first column

    def __len__(self):
        return len(self.patient_images)

    def __getitem__(self, idx):
        image_paths = self.patient_images[idx]
        patient_id = self.patient_ids[idx]
        raw_gene_value = self.gene_values[idx]
        raw_gene_value = float(np.nan_to_num(raw_gene_value, nan=0.0))
        class_label = zscore_to_class(raw_gene_value)
        gene_values = torch.tensor(class_label, dtype=torch.long)
        corresponding_genes = self.corresponding_gene

        images = []

        for path in image_paths:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)

                # Normalize image
                img -= img.min()
                if img.max() != 0:
                    img /= img.max()

                if img.ndim == 3:
                    # Volume: [D, H, W] → treat as multiple slices
                    for slice_img in img:
                        slice_tensor = torch.tensor(slice_img).unsqueeze(0)  # [1, H, W]
                        if self.transform:
                            slice_tensor = self.transform(slice_tensor)
                        images.append(slice_tensor)

                elif img.ndim == 2:
                    # Single slice: [H, W]
                    slice_tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
                    if self.transform:
                        slice_tensor = self.transform(slice_tensor)
                    images.append(slice_tensor)

                else:
                    print(f"⚠️ Unexpected shape {img.shape} in file {path} — skipping")

            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")

        return images, patient_id, gene_values, corresponding_genes
class PatientDicomDatasetSingleGeneByIndexClassBinary(Dataset):

    def __init__(self, root_dir, csv_path,index, transform=None):
        """takes an index and extracts the relevant gene from the csv file"""
        #print("Starting dataset load")
        self.transform = transform
        self.patient_images = []
        self.patient_ids = []
        self.gene_values = []
        print("Loading CSV and images...")
        # Load CSV
        df = pd.read_csv(csv_path, dtype=str)
        df.set_index(df.columns[0], inplace=True)  # Set first column (genes) as index
        df = df.transpose()  # Now rows are patients, columns are genes
        # now slice off the single gene
        df = df.iloc[:, index:index+1]  # Get only the specified gene column
        df.index.name = 'patient_id'
        # store the corresponding gene name as a string, not a list
        self.corresponding_gene = df.columns[0]
        
        # print("CSV loaded successfully.")
        # print("Number of patients in CSV:", len(df))
        # Collect images per patient
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
           
            if not os.path.isdir(patient_path):
                continue
            #add all images in the patient folder
            #gonna do away with metadata for positionality and such for now
            images = []
            for body_part_folder in os.listdir(patient_path):
                body_part_path = os.path.join(patient_path, body_part_folder)
                if not os.path.isdir(body_part_path):
                    continue
                for image_folder in os.listdir(body_part_path):
                    image_path = os.path.join(body_part_path, image_folder)
                    if not os.path.isdir(image_path):
                        continue

                    dcm_files = [f for f in os.listdir(image_path) if f.endswith('.dcm')]
                    dcm_files = sorted(dcm_files, key=extract_index_number)

                    # Keep only the 32 most central slices
                    if len(dcm_files) > 8:
                        center = len(dcm_files) // 2
                        dcm_files = dcm_files[center - 16 : center + 16]

                    for file in dcm_files:
                        full_path = os.path.join(image_path, file)
                        images.append(full_path)
            if patient_id in df.index:
                # print(images)
                gene_val= df.loc[patient_id].values[0]
                #if nan then skip over it and delete the images added to path
                if pd.isna(gene_val):
                    print(f"⚠️ Skipping patient {patient_id} due to NaN gene value for {self.corresponding_gene}")
                    #need to clear images of this patient
                    images.clear()
                    continue
                self.gene_values.append(gene_val)
                
                self.patient_images.append(images.copy())
                self.patient_ids.append(patient_id)
                images.clear()
                
                
                #want to get the first row minus the first column

    def __len__(self):
        return len(self.patient_images)

    def __getitem__(self, idx):
        image_paths = self.patient_images[idx]
        patient_id = self.patient_ids[idx]
        raw_gene_value = self.gene_values[idx]
        raw_gene_value = float(np.nan_to_num(raw_gene_value, nan=0.0))
        class_label = zscore_to_class_binary(raw_gene_value)
        gene_values = torch.tensor(class_label, dtype=torch.long)
        corresponding_genes = self.corresponding_gene

        images = []

        for path in image_paths:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)

                # Normalize image
                img -= img.min()
                if img.max() != 0:
                    img /= img.max()

                if img.ndim == 3:
                    # Volume: [D, H, W] → treat as multiple slices
                    for slice_img in img:
                        slice_tensor = torch.tensor(slice_img).unsqueeze(0)  # [1, H, W]
                        if self.transform:
                            slice_tensor = self.transform(slice_tensor)
                        images.append(slice_tensor)

                elif img.ndim == 2:
                    # Single slice: [H, W]
                    slice_tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
                    if self.transform:
                        slice_tensor = self.transform(slice_tensor)
                    images.append(slice_tensor)

                else:
                    print(f"⚠️ Unexpected shape {img.shape} in file {path} — skipping")

            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")

        return images, patient_id, gene_values, corresponding_genes

def zscore_to_class(z, pos_thresh=1.0, neg_thresh=-1.0):
    """
    Convert z-score to classification label based
    the thresholds for overexpression and underexpression are about the same"""
    if z >= pos_thresh:
        return 2  # overexpressed
    elif z <= neg_thresh:
        return 0  # underexpressed
    else:
        return 1  # normal
    

def zscore_to_class_binary(z, pos_thresh=1.0, neg_thresh=-1.0):
    """
    Convert z-score to classification label based
    the thresholds for overexpression and underexpression are about the same"""
    if z >= 0:
        return 1  # overexpressed
    elif z < 0:
        return 0  # underexpressed
