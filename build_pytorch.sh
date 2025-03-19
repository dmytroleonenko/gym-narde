#!/bin/bash
set -e -x

# Check that we're using Python 3.13t
PYTHON_PATH="/opt/homebrew/Caskroom/miniconda/base/envs/py313t/bin/python"
$PYTHON_PATH -c "import sys; print('Using Python:', sys.version); print('Python path:', sys.executable); import sysconfig; print('GIL Disabled:', bool(sysconfig.get_config_var(\"Py_GIL_DISABLED\"))); print('Architecture:', platform.machine())" 2>/dev/null || $PYTHON_PATH -c "import sys, platform; print('Using Python:', sys.version); print('Python path:', sys.executable); import sysconfig; print('GIL Disabled:', bool(sysconfig.get_config_var(\"Py_GIL_DISABLED\"))); print('Architecture:', platform.machine())"

# Install dependencies
$PYTHON_PATH -m pip install -U pip setuptools wheel
$PYTHON_PATH -m pip install -U cmake ninja numpy pyyaml requests pkgconfig
$PYTHON_PATH -m pip install -U "typing-extensions>=4.10.0"
$PYTHON_PATH -m pip install -U filelock jinja2 fsspec "sympy>=1.13.3"
$PYTHON_PATH -m pip install -U optree networkx "expecttest>=0.3.0"

# Go to PyTorch directory
cd pytorch

# Fix the import issues that are Python 3.13 specific
for file in $(grep -l "from typing_extensions import Self" --include="*.py" -r .); do
    echo "Patching $file"
    sed -i.bak 's/from typing_extensions import Self/try:\n    from typing import Self\nexcept ImportError:\n    from typing_extensions import Self/' "$file"
done

# Modify setup.py to add Python 3.13 to the list of supported versions
sed -i.bak 's/python_requires=">=3.8",/python_requires=">=3.8",\n    classifiers=[\n        "Programming Language :: Python :: 3.8",\n        "Programming Language :: Python :: 3.9",\n        "Programming Language :: Python :: 3.10",\n        "Programming Language :: Python :: 3.11",\n        "Programming Language :: Python :: 3.12",\n        "Programming Language :: Python :: 3.13",\n    ],/' setup.py

# Set environment variables for build
export CMAKE_PREFIX_PATH=/opt/homebrew/Caskroom/miniconda/base/envs/py313t/
export PYTHON_EXECUTABLE=$PYTHON_PATH
export BUILD_TEST=0
export USE_CUDA=0
export USE_XPU=0 
export USE_DISTRIBUTED=0
export USE_NUMPY=1
export USE_MKLDNN=0
export USE_OPENMP=0
export MAX_JOBS=4
export MACOSX_DEPLOYMENT_TARGET=11.0
unset CC
unset CXX
export CFLAGS="-DPYTHON_COMPAT_3_13=1"
export CXXFLAGS="-DPYTHON_COMPAT_3_13=1"

# Clean any previous build artifacts
rm -rf build dist torch.egg-info

# Build PyTorch with the simplified approach
$PYTHON_PATH -m pip install -e .

# Verify installation
cd ..
$PYTHON_PATH -c "import torch; print('PyTorch version:', torch.__version__); print('Built with:', torch.__config__.show())" 