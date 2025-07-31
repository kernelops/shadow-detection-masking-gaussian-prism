#!/usr/bin/env python3
"""
Test script to verify installation of shadow detection and removal project
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package=package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
            return True
        else:
            print("⚠ CUDA is not available. Will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(test_img, (5, 5), 0)
        print("✓ OpenCV basic operations work")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Shadow Detection and Removal Installation")
    print("=" * 50)
    
    # Test core dependencies
    print("\n1. Testing Core Dependencies:")
    print("-" * 30)
    
    dependencies = [
        "numpy",
        "cv2",
        "torch",
        "torchvision",
        "skimage",
        "scipy",
        "matplotlib"
    ]
    
    all_passed = True
    for dep in dependencies:
        if not test_import(dep):
            all_passed = False
    
    # Test CUDA
    print("\n2. Testing CUDA:")
    print("-" * 30)
    test_cuda()
    
    # Test OpenCV
    print("\n3. Testing OpenCV:")
    print("-" * 30)
    if not test_opencv():
        all_passed = False
    
    # Test project files
    print("\n4. Testing Project Files:")
    print("-" * 30)
    
    import os
    required_files = [
        "main.py",
        "shadow_detect.py", 
        "shadow_remove.py",
        "segment.py",
        "shadow.jpg"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} not found")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Installation is successful.")
        print("\nTo run the project:")
        print("  python main.py")
        print("\nOr use the improved version:")
        print("  python main_improved.py")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 