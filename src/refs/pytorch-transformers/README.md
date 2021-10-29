# What is this?
This is a fork of the PyTorch-Transformers repo for the FEVER symmetric dataset processor and weighted loss.
For more details, see the [FeverSymmetric](https://github.com/TalSchuster/FeverSymmetric) repository.

## Training

```
python examples/run_glue.py \
  --task_name fever \
  --do_train \
  --do_eval \
  --do_lower_case \
  --model_type bert \
  --data_dir PATH_TO_DATA_DIR \
  --model_name_or_path bert-base-uncased \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --save_steps 100000 \
  --output_dir output/baseline
```

to use the per sample weights, use the `--weighted_loss` flag.

## Testing

```
python examples/run_glue.py \
  --task_name fever \
  --do_eval \
  --output_preds \
  --do_lower_case \
  --model_type bert \
  --data_dir PATH_TO_DATA_DIR \
  --model_name_or_path bert-base-uncased \
  --max_seq_length 128 \
  --output_dir output/baseline
```
