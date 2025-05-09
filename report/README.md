# Interior Design Style Transfer Project

## 1. Project Overview

### 1.1 Project Objective

The primary goal of this project is to develop an interior design style transfer system that can transform rooms from one design style to another while preserving structural elements. Using the optimized CycleGAN architecture from Fu (2022), we aim to:

1. Create a robust style transfer model that can convert between different interior design styles (modern, minimalist, industrial, boho, and scandinavian)
2. Preserve important structural elements (walls, windows, architectural features) while transforming decorative elements
3. Develop a computationally efficient implementation that reduces resource requirements compared to standard CycleGAN
4. Provide an intuitive tool that allows designers and homeowners to visualize spaces in different styles without requiring extensive manual redesign

This technology enables rapid prototyping of interior design concepts and allows for exploration of style alternatives that might otherwise be difficult to visualize. The project focuses initially on single room type transformation (living rooms) with plans to expand to other room types as the model is refined.

### 1.2 Research Methodology

This project is based on the research paper:

**Digital Image Art Style Transfer Algorithm Based on CycleGAN**

- Author: Xuhui Fu
- Published: January 13, 2022
- DOI: [https://doi.org/10.1155/2022/6075398](https://doi.org/10.1155/2022/6075398)
- Part of Special Issue: Computational Intelligence in Image and Video Analysis

The paper presents an optimized CycleGAN architecture specifically designed for artistic style transfer of images. The improvements focus on:

1. Reducing computational resource requirements while maintaining quality
2. Enhancing the preservation of structural elements during style transformation
3. Optimizing the generator and discriminator networks for better style transfer results
4. Implementing a more efficient training process for faster convergence

## 2. Dataset Information

### 2.1 Dataset Overview

This dataset contains a curated collection of interior design images categorized by room type and design style. The images are sourced from Pinterest and labeled with relevant metadata for machine learning applications, including image classification, style prediction, style transfer, and aesthetic analysis.

### 2.2 Dataset Structure

The dataset is organized hierarchically in the following structure:

```
data/
├── raw/
│   ├── bathroom/
│   │   ├── boho/
│   │   │   ├── bathroom_boho_0.jpg
│   │   │   ├── bathroom_boho_1.jpg
│   │   │   └── ...
│   │   ├── industrial/
│   │   ├── minimalist/
│   │   ├── modern/
│   │   └── scandinavian/
│   ├── bedroom/
│   │   ├── boho/
│   │   ├── industrial/
│   │   ├── minimalist/
│   │   ├── modern/
│   │   └── scandinavian/
│   ├── kitchen/
│   │   ├── boho/
│   │   ├── industrial/
│   │   ├── minimalist/
│   │   ├── modern/
│   │   └── scandinavian/
│   └── living_room/
│       ├── boho/
│       ├── industrial/
│       ├── minimalist/
│       ├── modern/
│       └── scandinavian/
├── metadata.csv
├── train_data.csv
├── val_data.csv
└── test_data.csv
```

### 2.3 Room Types

The dataset includes images from four primary room types:

- `bathroom`: Bathroom interior designs
- `bedroom`: Bedroom interior designs
- `kitchen`: Kitchen interior designs
- `living_room`: Living room interior designs

### 2.4 Design Styles

Each room type contains examples of five different interior design styles:

- `boho`: Bohemian style featuring eclectic, global influences with casual and layered aesthetics
- `industrial`: Raw, utilitarian style with exposed materials, metal fixtures, and minimally finished surfaces
- `minimalist`: Clean, uncluttered spaces with simple color palettes and essential furniture
- `modern`: Contemporary designs with clean lines, updated materials, and functional aesthetics
- `scandinavian`: Light, airy spaces with natural materials, neutral colors, and functional simplicity

### 2.5 CSV Files and Metadata

#### 2.5.1 metadata.csv

Contains the complete dataset metadata with the following columns:

- `image_path`: Relative path to the image file (e.g., `../data/raw/bathroom/boho/bathroom_boho_0.jpg`)
- `room_type`: The category of room (one of: bathroom, bedroom, kitchen, living_room)
- `style`: The design style (one of: boho, industrial, minimalist, modern, scandinavian)

#### 2.5.2 train_data.csv

Contains the training split of the dataset with the following columns:

- `image_path`: Relative path to the image file
- `room_type`: The category of room
- `style`: The design style
- `encoded_label`: Numeric encoding of the combined room type and style (useful for classification tasks)

#### 2.5.3 val_data.csv

Contains the validation split of the dataset with the same column structure as metadata.csv.

#### 2.5.4 test_data.csv

Contains the test split of the dataset with the same column structure as metadata.csv.

#### 2.5.5 Encoded Labels

The `encoded_label` column in train_data.csv uses numeric encoding for the combination of room type and style:

- Values for bathroom styles: 0-4
- Values for bedroom styles: 5-9
- Values for kitchen styles: 10-14
- Values for living_room styles: 15-19

Within each room type, the styles are encoded in alphabetical order (boho, industrial, minimalist, modern, scandinavian).

## 3. Working with the Dataset

### 3.1 Loading with kagglehub API

The dataset can be accessed directly using the kagglehub API, which is the recommended approach:

```python
# Install dependencies as needed:
# pip install kagglehub
# pip install kagglehub[pandas-datasets]  # for pandas adapter

# Method 1: Download the entire dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("galinakg/interior-design-images-and-metadata")
print("Path to dataset files:", path)

# Method 2: Load a specific CSV file directly into pandas
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load (e.g., "metadata.csv", "train_data.csv", etc.)
file_path = "train_data.csv"  # or "metadata.csv", "val_data.csv", "test_data.csv"

# Load the specified file into a DataFrame
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "galinakg/interior-design-images-and-metadata",
  file_path
)
print("First 5 records:", df.head())
```

### 3.2 Loading with pandas (if files already downloaded)

If you've already downloaded the dataset files to your local environment:

```python
import pandas as pd

# Load the full metadata
metadata = pd.read_csv('metadata.csv')

# Load specific splits
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

# Example: Filter for specific room type and style
modern_kitchens = metadata[(metadata['room_type'] == 'kitchen') &
                          (metadata['style'] == 'modern')]
```

### 3.3 Data Preparation for CycleGAN

For unpaired image-to-image translation tasks like style transfer with CycleGAN, you'll want to organize images by domain. Following the methodology from the Fu (2022) paper on optimized CycleGAN for style transfer:

```python
import os
import shutil

# Create directories for CycleGAN training
os.makedirs('cyclegan_data/trainA', exist_ok=True)  # Source style
os.makedirs('cyclegan_data/trainB', exist_ok=True)  # Target style
os.makedirs('cyclegan_data/testA', exist_ok=True)   # For testing source -> target
os.makedirs('cyclegan_data/testB', exist_ok=True)   # For testing target -> source

# Example: Transform from modern to minimalist living rooms
source_style = train_data[(train_data['room_type'] == 'living_room') &
                         (train_data['style'] == 'modern')]

target_style = train_data[(train_data['room_type'] == 'living_room') &
                         (train_data['style'] == 'minimalist')]

# Test data
source_style_test = test_data[(test_data['room_type'] == 'living_room') &
                             (test_data['style'] == 'modern')]

target_style_test = test_data[(test_data['room_type'] == 'living_room') &
                             (test_data['style'] == 'minimalist')]

# Copy training images to appropriate directories
for _, row in source_style.iterrows():
    src_path = row['image_path']
    # Remove the leading '../' if necessary
    if src_path.startswith('../'):
        src_path = src_path[3:]
    dst_path = os.path.join('cyclegan_data/trainA', os.path.basename(src_path))
    shutil.copy(src_path, dst_path)

for _, row in target_style.iterrows():
    src_path = row['image_path']
    if src_path.startswith('../'):
        src_path = src_path[3:]
    dst_path = os.path.join('cyclegan_data/trainB', os.path.basename(src_path))
    shutil.copy(src_path, dst_path)

# Copy test images
for _, row in source_style_test.iterrows():
    src_path = row['image_path']
    if src_path.startswith('../'):
        src_path = src_path[3:]
    dst_path = os.path.join('cyclegan_data/testA', os.path.basename(src_path))
    shutil.copy(src_path, dst_path)

for _, row in target_style_test.iterrows():
    src_path = row['image_path']
    if src_path.startswith('../'):
        src_path = src_path[3:]
    dst_path = os.path.join('cyclegan_data/testB', os.path.basename(src_path))
    shutil.copy(src_path, dst_path)
```

## 4. Implementation Approach

### 4.1 CycleGAN Architecture

Our implementation will follow the improved CycleGAN architecture from Fu (2022) with:

1. **Enhanced Generator Network**:

   - Residual blocks for better feature preservation
   - Improved downsampling/upsampling to maintain spatial information

2. **Multi-scale Discriminator**:

   - Focuses on local patches at different scales
   - Better preservation of texture and detail

3. **Custom Loss Functions**:
   - Cycle consistency loss to ensure content preservation
   - Structural preservation loss to maintain architectural elements
   - Style transfer loss for aesthetic transformation

### 4.2 Training Process

The training will follow these steps:

1. Data preparation as outlined in section 3.3
2. Optimized training procedures from the Fu (2022) paper:

   - Progressive learning rate scheduling
   - Custom batch size and iteration count
   - Early stopping based on validation metrics

3. Monitoring both qualitative results (visual inspection) and quantitative metrics:
   - FID (Fréchet Inception Distance) for measuring image quality
   - LPIPS (Learned Perceptual Image Patch Similarity) for perceptual similarity

### 4.3 Evaluation

The evaluation will use:

1. The test_data.csv split for objective assessment
2. Comparison against baseline CycleGAN implementations
3. Metrics to assess both style transfer quality and structural preservation
4. User studies for subjective quality assessment (optional)

## 5. Additional Information

### 5.1 Use Cases

This dataset and model implementation are well-suited for various applications:

1. **Interior Design Visualization**: Allow designers to show clients various style options
2. **Real Estate Staging**: Virtually redecorate spaces for marketing
3. **Home Renovation Planning**: Help homeowners visualize different design approaches
4. **E-commerce**: Show furniture in different style contexts
5. **Research**: Advance the field of conditional image generation and style transfer

### 5.2 Citation

When using this dataset and methodology in your research or projects, please cite:

For the dataset:
"Interior Design Images and Metadata Dataset from Pinterest"

For the methodology:
Fu, X. (2022). Digital Image Art Style Transfer Algorithm Based on CycleGAN. Computational Intelligence and Neuroscience, 2022. https://doi.org/10.1155/2022/6075398

### 5.3 License

[Include license information if available]
