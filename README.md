# GPU-Research


Steps to replicate the experiments:
1. [Mount onto Docker]
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/GPU-Research:/code nvcr.io/nvidia/pytorch:21.10-py3
cd /code
pip install -e .
3. 
