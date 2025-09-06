#!/bin/bash

#!/bin/bash

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
git clone https://github.com/LakshyaSingh354/CudaCode.git
# git clone https://github.com/LakshyaSingh354/Inferno.git
# cd Inferno
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
git config --global user.email "lakshya.singh354@gmail.com"
git config --global user.name "Lakshya Singh"