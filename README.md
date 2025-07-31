# Shadow Detection and Removal

Digital Image processing - NITT course project

## Overview

Reimplementation of the paper [Efficient Shadow Removal Using Subregion Matching Illumination Transfer](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12250?casa_token=XaNTua352PwAAAAA%3AbThsqn8IUYmAwvGgR0-iVLmKTn8SI0YaYy1APLNI1hbzCpHLyakUAGy1ICcWy4YgvJCD2vvlaMeEH_Zt "Efficient Shadow Removal Using Subregion Matching Illumination Transfer") with a slight twist.

Implemented an unsupervised segmentation algorithm employing autoencoders for detection of shadow regions. Gabor filter is designed to identify the texture features in the images. Illuminance transfer techniques are deployed to remove the shadow regions with the help of the acquired textures.

Implemented the algorithm on SBU Shadow Datasets obtaining good results. More work is needed on the boundary processing of the shadows to improve the accuracy.

![gesture detection](flow.png)

The flow of the shadow detection and removal algorithm

## Features

- **Shadow Detection**: Uses HSI color space conversion and Otsu's thresholding method
- **Unsupervised Segmentation**: CNN-based segmentation using PyTorch
- **Gabor Filtering**: Multi-scale and multi-orientation texture feature extraction
- **Subregion Matching**: Texture and spatial distance-based matching
- **Illumination Transfer**: Shadow removal using matched regions

## Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd shadow_detection_and_removal

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using conda

```bash
# Create a new conda environment
conda create -n shadow-detection python=3.8
conda activate shadow-detection

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install opencv-python numpy scikit-image scipy matplotlib
```

### Method 3: Manual Installation

```bash
# Install PyTorch first (visit pytorch.org for your specific setup)
pip install torch torchvision

# Install other dependencies
pip install opencv-python numpy scikit-image scipy matplotlib
```

## Usage

### Quick Start

1. **Place your input image** as `shadow.jpg` in the project directory
2. **Run the main script**:
   ```bash
   python main.py
   ```

### Individual Components

You can also run individual components:

- **Shadow Detection Only**:
  ```bash
  python shadow_detect.py
  ```

- **Shadow Removal Only**:
  ```bash
  python shadow_remove.py
  ```

- **Segmentation Only**:
  ```bash
  python segment.py
  ```

### Custom Input Image

To use a different input image, modify the `input_file` variable in `main.py`:

```python
input_file = 'your_image.jpg'  # Change this line
```

## Output

The program will display several windows:
- **Original Image**: The input image with shadows
- **Detected Shadow**: Binary mask showing detected shadow regions
- **Shadow Region**: Segmented shadow areas
- **Background Region**: Segmented non-shadow areas  
- **Shadow Removed Image**: Final result with shadows removed

Press any key to close the windows.

## Project Structure

```
shadow_detection_and_removal/
├── main.py              # Main script (complete pipeline)
├── shadow_detect.py     # Shadow detection component
├── shadow_remove.py     # Shadow removal component
├── segment.py           # Segmentation component
├── gabor_filter.py      # Gabor filter implementation
├── gabor.py             # Gabor filter utilities
├── final.py             # Alternative implementation
├── requirements.txt     # Python dependencies
├── setup.py            # Installation script
├── shadow.jpg          # Sample input image
├── output.png          # Sample output
├── flow.png            # Algorithm flowchart
└── README.md           # This file
```

## Algorithm Overview

1. **Preprocessing**: Bilateral filtering for noise reduction
2. **HSI Conversion**: Convert to HSI color space for better shadow detection
3. **Shadow Detection**: Otsu's thresholding on ratio map
4. **Segmentation**: CNN-based unsupervised segmentation
5. **Feature Extraction**: Gabor filtering for texture features
6. **Subregion Matching**: Match shadow and background regions
7. **Illumination Transfer**: Remove shadows using matched regions

## Parameters

Key parameters in `main.py`:
- `nChannel`: Number of segmentation channels (default: 100)
- `maxIter`: Maximum iterations for segmentation (default: 100)
- `minLabels`: Minimum number of labels (default: 3)
- `lr`: Learning rate (default: 0.1)
- `num_superpixels`: Number of superpixels (default: 10000)
- `compactness`: SLIC compactness (default: 100)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `num_superpixels` or image size
2. **Import errors**: Ensure all dependencies are installed
3. **OpenCV display issues**: Use a different backend or run on a machine with display

### Performance Tips

- Use GPU if available (CUDA)
- Reduce image size for faster processing
- Adjust `maxIter` and `num_superpixels` based on your needs

## Reference

1. [Efficient Shadow Removal Using Subregion Matching Illumination Transfer](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.12250?casa_token=XaNTua352PwAAAAA%3AbThsqn8IUYmAwvGgR0-iVLmKTn8SI0YaYy1APLNI1hbzCpHLyakUAGy1ICcWy4YgvJCD2vvlaMeEH_Zt "Efficient Shadow Removal Using Subregion Matching Illumination Transfer")
2. [Unsupervised Segmentation](https://github.com/kanezaki/pytorch-unsupervised-segmentation "Unsupervised Segmentation")

## Authors

- Bharath Kumar
- Chockalingam  
- DC Vivek
- Harinath Gobi

## License

This project is for educational purposes.
