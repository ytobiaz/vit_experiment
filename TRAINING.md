![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# Training Instructions

The instructions have been primarily adopted from [NViT](https://github.com/NVlabs/NViT).

## Pruning

`main.py` can be used to perform global structural pruning on a pretrained DeiT-B model towards a target speedup. The pretrained DeiT-B model checkpoint will be downloaded automatically.

Arguments:

- `epochs` - maximum epochs of pruning, can be set to a large number, code will stop when target latency is reached
- `data-path` - path to ImageNet dataset
- `output_dir` - path to save pruning log and final pruned model. Important arguments will be automatically appended to the specified folder name
- `lr` - learning rate for model update during iterative pruning
- `prune_per_iter` - number of neurons pruned in each pruning iteration
- `interval_prune` - number of optimization steps (batches) within each pruning iteration
- `latency_regularization` - weight of latency regularization in overall sensitivity 
- `latency_target` - ratio of the target latency of the final pruned model and the latency of the original model

The pruning process will stop after the target latency is reached. The model after pruning is stored in the `pruned_checkpoint.pth`, and the pruning log and remaining dimension of each pruning step can be found in the `debug` folder.

The following arguments were used to prune the Base model. For the other models, the arguments remain the same except for a change in the latency target.

```
 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_full_global_latency.py --model deit_base_distilled_patch16_224 --epochs 50 --num_workers 10 --batch-size 128 --data-path /path/to/ImageNet2012/ --data-set IMNET --lr 1e-4 --output_dir save/path --amp --input-size 224 --seed 1 --pruning_config=pruning_configs/group8_m23_m09.json --prune_per_iter=32 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --student_eval=True --dist-eval --pruning --prune_dict '{"Global":39000}' --interval_prune 100 --pretrained --distillation-type hard --latency_regularization 5e-4 --latency_target 0.54 --latency_look_up_table latency_head.json --pruning_exit
```

## Fine-tune on ImageNet

After pruning, `finetune_dense.py` can be used to convert the pruned model into a small dense model, and perform fine-tuning.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune_dense.py --model deit_base_distilled_patch16_224 --epochs 300 --num_workers 10 --batch-size 144 --data-path /path/to/ImageNet2012/ --data-set IMNET --amp --input-size 224 --seed 1 --kl_loss_coeff=100000 --original_loss_coeff=1.0 --dist-eval --pretrained --finetune path/to/pruned_model --distillation-type hard --distillation-alpha 0.5 --distillation-tau 20.0 --lr 0.0002
```

## Fine-tune on Downstream Tasks

After fine-tuning on ImageNet, `finetune_tl.py` can be used to apply transfer learning to the fine-tuned dense model.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune_dense.py --model deit_base_distilled_patch16_224 --epochs 300 --num_workers 10 --batch-size 128 --data-path /path/to/dataset/ --data-set dataset --amp --input-size 224 --seed 1 --lr 0.0002
```
