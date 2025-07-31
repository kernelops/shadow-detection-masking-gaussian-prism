# 3D-GS-Setup
Procedure and dependency versions for the full 3D-GS offical repository [Model Train + View] setup on Windows.

# Prerequisites

- __Git__ - [here ](https://git-scm.com/downloads). Follow default installation instructions. You can test to see if you have it already installed by typing ```git --version``` into command prompt
- __Conda__ - [Anaconda](https://www.anaconda.com/download) 
- __CUDA Toolkit__ -  11.8 [here](https://developer.nvidia.com/cuda-toolkit-archive). You can check which version of CUDA Toolkit you have installed by typing ```nvcc --version``` into command prompt.
- __Visual Studio__ 2019 [here](https://www.techspot.com/downloads/downloadnow/7241/?evp=70f51271955e6392571f575e301cd9a3&file=9642). Make sure you add __Desktop Development with C++__ when installing
- __COLMAP__ - [here](https://github.com/colmap/colmap/releases)
- __ImageMagik__ - [here](https://imagemagick.org/script/download.php)
- __FFMPEG__ - [here](https://ffmpeg.org/download.html)
- __Nvidia GPU and up-to-date Drivers__

# 1. Clone the Repository

Run Powershell as administrator

```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
``` 

# 2. Set up the conda environment 

cd into the repo folder where environment.yml is present 

```shell
SET DISTUTILS_USE_SDK=1                         #On windows
```
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

In case there are dependency errors and the env creation stops in the middle, delete the environment, fix dependencies and re-run. 

```shell 
conda env remove -n gaussian_splatting
```
[Try pip install ninja]

# 3. Train a model 

Download the pre-processed ready to train T&T+DB COLMAP (650MB) dataset [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

Extract this tandt_db.zip file in a new data folder. Then run

```shell
python train.py -s data/tandt/truck --iterations 7000    #default is 30000
```

At the end of training it puts all files in a new output folder. 


# 4. Set up the viewer

**Download the SIBR Pre-built Viewers**

  Download the viewers from the official 3D Gaussian Splatting project:
  [zip file](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)

  Extract the Zip File in a new folder.
    
# 5. Run the viewer 

1.  **Open PowerShell or Command Prompt.**
2.  **Navigate to the `bin` directory of the extracted SIBR viewers:**
    ```powershell
    cd "C:\Path\To\Your\Extracted\Viewers\bin"
    ```

3.  **Run the viewer application, pointing it to the trained model directory:**
    ```powershell
    .\SIBR_gaussianViewer_app.exe -m "C:\Path\To\Your\ae542f1e-6 or folder name"
    ```

# 6. Navigation Controls

*   A new window should open displaying the 3D scene.
*   **Navigation Controls:**
    *   **Translate Camera:** `W` (forward), `A` (left), `S` (backward), `D` (right), `Q` (down), `E` (up)
    *   **Rotate Camera:** `I` (pitch up), `K` (pitch down), `J` (yaw left), `L` (yaw right), `U` (roll left), `O` (roll right)
*   Explore the floating menus for other options like changing navigation speed or visual settings.
*   **Performance Tip:** For smoother frame rates, disable V-Sync in your NVIDIA Control Panel, and in the viewer application (usually under a "Display" or "Render" menu).
