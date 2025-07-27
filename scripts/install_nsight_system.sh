#!/bin/bash

set -e

if command -v nsys &> /dev/null; then
    echo "NVIDIA Nsight Systems (nsys) is already installed:"
    nsys --version
    exit 0
fi


apt update
wget https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
apt install ./NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
nsys --version
rm NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb