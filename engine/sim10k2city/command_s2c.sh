# If -debug, no wandb records experiment data
# Source domain data:sim10k    Target domain data:cityscapes

# test command
python -m torch.distributed.run --nproc_per_node=2 \
  main.py -m dab_deformable_detr \
  --output_dir logs/dab_deformable_detr/R50 \
  --batch_size 3 \
  --eval \
  --resume checkpoint/sim10k2city/tea_76.83.pth \
  --coco_path /path/to/dataset \
  --da_task sim10k2city \
  --debug

# train command

# NO pretrain   mAP:57.59%
python -m torch.distributed.run --nproc_per_node=2 \
  main.py -m dab_deformable_detr \
  --batch_size 2 \
  --epochs 20 \
  --lr 0.00014 \
  --lr_drop 20 \
  --con_weight 2.0 \
  --dis_weight 1.0 \
  --model_ema_decay 0.999 \
  --coco_path /path/to/dataset \
  --warm_epochs 12 \
  --cos_thre 0.89 \
  --top_num 2 \
  --low_iou 0.9 \
  --high_iou 0.3 \
  --last_iou 0.2 \
  --da_task sim10k2city
