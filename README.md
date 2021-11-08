# GPU-Research


Steps to replicate the experiments:

[Mount onto Docker]

1. docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/GPU-Research:/code nvcr.io/nvidia/pytorch:21.10-py3
2. cd /code
3. pip install -e .


