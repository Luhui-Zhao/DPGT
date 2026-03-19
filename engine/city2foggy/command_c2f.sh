# If -debug, no wandb records experiment data
# Source domain data:cityscapes    Target domain data:Foggy cityscapes

# test command
python -m torch.distributed.run --nproc_per_node=2 \
  main.py -m dab_deformable_detr \
  --output_dir logs/dab_deformable_detr/R50 \
  --batch_size 3 \
  --eval \
  --resume checkpoint/city2foggy/tea_62.05.pth \
  --coco_path /path/to/dataset \
  --da_task city2foggy \
  --debug

# train command
# pretrain   mAP:62.05%
python -m torch.distributed.run --nproc_per_node=2 \
  main.py -m dab_deformable_detr \
  --batch_size 2 \
  --epochs 22 \
  --lr 0.00014 \
  --lr_drop 20 \
  --con_weight 2.0 \
  --dis_weight 2.0 \
  --model_ema_decay 0.999 \
  --pretrain_model_path checkpoint/DAB-Deformable-DETR-R50-v2.pth \
  --coco_path /path/to/dataset \
  --warm_epochs 8 \
  --cos_thre 0.89 \
  --top_num 2 \
  --low_iou 0.9 \
  --high_iou 0.3 \
  --last_iou 0.2 \
  --da_task city2foggy

# NO pretrain   mAP:53.52%
python -m torch.distributed.run --nproc_per_node=2 \
  main.py -m dab_deformable_detr \
  --batch_size 2 \
  --epochs 35 \
  --lr 0.00014 \
  --lr_drop 33 \
  --con_weight 2.0 \
  --dis_weight 1.0 \
  --model_ema_decay 0.999 \
  --coco_path /path/to/dataset \
  --warm_epochs 22 \
  --cos_thre 0.89 \
  --top_num 2 \
  --low_iou 0.9 \
  --high_iou 0.30 \
  --last_iou 0.2 \
  --da_task city2foggy
