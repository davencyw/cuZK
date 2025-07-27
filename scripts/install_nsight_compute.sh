#!/bin/bash

set -e

if command -v ncu &> /dev/null; then
    echo "NVIDIA Nsight Compute (ncu) is already installed:"
    ncu --version
    exit 0
fi

echo "Installing NVIDIA Nsight Compute..."
apt update
apt install -y nsight-compute

echo "Verifying installation..."
ncu --version

echo "✓ NVIDIA Nsight Compute installed successfully!"
echo "Command: ncu" 