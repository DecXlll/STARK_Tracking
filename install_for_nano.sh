##for jetson nano - jetpack 4.6
echo "export CUBA_HOME=/usr/local/cuda-10.2">> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-10.2/bin:$PATH" >> ~/.bashrc
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
echo "alias pipenv='python3 -m pipenv'" >> ~/.bashrc
source ~/.bashrc

sudo apt install llvm-9
LLVM_CONFIG=llvm-config-9 pip install llvmlite==0.33
echo "alias llvm-config='llvm-config-9'" >> ~/.bashrc
echo "export LLVM_CONFIG='/usr/bin/llvm-config-9'" >> ~/.bashrc
source ~/.bashrc



#pip install torch==1.8.0
pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl


##install torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev libopenblas-base libopenmpi-dev
#git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision   
#cd torchvision
#export BUILD_VERSION=0.9.0 
#python3 setup.py install --user
#cd ../  
pip install 'pillow<7'


pip install torchvision==0.9.1


pip install mmcv-full==1.5.3
pip install mmdet
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .
