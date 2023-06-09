sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev libopenblas-base libopenmpi-dev

##for jetson nano - jetpack 5.0.2
echo "export CUDA_HOME=/usr/local/cuda-11.4">> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-11.4/bin:$PATH" >> ~/.bashrc
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
echo "alias pipenv='python3 -m pipenv'" >> ~/.bashrc
source ~/.bashrc

sudo apt install llvm-9
LLVM_CONFIG=llvm-config-9 pip install llvmlite==0.33
echo "alias llvm-config='llvm-config-9'" >> ~/.bashrc
echo "export LLVM_CONFIG='/usr/bin/llvm-config-9'" >> ~/.bashrc
source ~/.bashrc

pipenv shell

pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

##install torchvision
#git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision   
#cd torchvision
#export BUILD_VERSION=0.9.0 
#python3 setup.py install --user
#cd ../  
pip install 'pillow<7'


pip install torchvision==0.13.0


pip install mmcv-full==1.5.3
pip install mmdet
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
