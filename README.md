# GPU-Research


Steps to replicate the experiments:


[Git Clone this Git Repository]
1. git clone https://github.com/JerryPan2718/GPU-Research.git


[Mount onto Docker]
1. docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/GPU-Research:/code nvcr.io/nvidia/pytorch:21.10-py3
2. cd /code
3. pip install -e .


[Download the data first]
1. cd ./flexgpt/
2. mkdir data
3. bash getdata.sh
4. Feel free to early abort the download of 1B words dataset after several seconds.


[Generate the model.pt with random parameter]
1. cd ./transformer-xl/
2. python3 train_random_model_params.py 


[Model Inference on the model.pt with customized setup]
1. Feel free to change all configs in "if __name__ == "__main__":" section: batch_size, tgt_len, ext_lens, mem_lens, clamp_len
2. python3 eval_random_model_params.py 


[After the above finishes running, check results]
1. cd ./logs/
2. View the log file you want.

 
