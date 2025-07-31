from setuptools import setup, find_packages

setup(
    name="shadow-detection-removal",
    version="1.0.0",
    description="Shadow Detection and Removal using Subregion Matching Illumination Transfer",
    author="Bharath Kumar, Chockalingam, DC Vivek, Harinath Gobi",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "scikit-image>=0.18.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.7",
) 