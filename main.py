# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import
import argparse
import datetime
import json
import time
from pathlib import Path
import os, sys
import wandb
import random
from util.logger import setup_logger
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_DABDETR, build_dab_deformable_detr
from util.utils import clean_state_dict
from util.utils import ModelEma
import pytz

# # for vscode debug
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)
    
    # about lr
    parser.add_argument('--lr', default=1.4e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-05, type=float, 
                        help='learning rate for backbone')
    #batch-size  
    # If GPU memory is below 24 GB, the batch size must be set to 1, and the hyperparameters should be adjusted accordingly.
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=22, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--save_checkpoint_interval', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m',default='dab_deformable_detr', type=str,  choices=['dab_detr', 'dab_deformable_detr'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--model_ema', default=True, action='store_true',help="To use the teacher model, must be true")
    parser.add_argument('--model_ema_decay', default=0.999, type=float,help="Teachers model the rate of EMA")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")

    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', 
                        help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='relu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in encoder layers")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Consistency loss weight coefficient
    parser.add_argument('--con_weight', default=2.0, type=float,help="Coefficient of consistency loss")
    # Distillation loss weight coefficient
    parser.add_argument('--dis_weight', default=1.0, type=float,help="Distillation loss coefficient")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float, 
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25, 
                        help="alpha for focal loss")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # dataset address
    parser.add_argument('--coco_path', default='',type=str,help="Data path")
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true', 
                        help="Using for debug only. It will fix the size of input images to the maximum.")
    parser.add_argument('--strong_aug', default=False)

    #prompt
    parser.add_argument('--prompt_pth_t', default='',help="Prompts for the target domain.")
    parser.add_argument('--prompt_pth_tf', default='',help="Prompt for false target domain")
    parser.add_argument('--open_pt', default=False,action='store_true',help="Enable prompt")
    parser.add_argument('--low_conf', default=0.0, type=float,help="Filter the minimum confidence of the query")
    parser.add_argument('--high_conf', default=1.0, type=float,help="Filter the highest confidence of the query")
    parser.add_argument('--cos_thre', default=0.89, type=float,help="Cosine similarity threshold")
    parser.add_argument('--learn_thre', default=0.5, type=float,help="The cosine similarity threshold used to update the hint")
    parser.add_argument('--top_num', default=2, type=int,help="At most, select several objects from an image to learn")
    parser.add_argument('--low_iou', default=0.9, type=float,help="Low confidence IOU threshold")
    parser.add_argument('--high_iou', default=0.3, type=float,help="High confidence IOU threshold")
    parser.add_argument('--last_iou', default=0.2, type=float,help="The IOU threshold for generating the final result")
    parser.add_argument('--pre_conf', default=0.5, type=float,help="The threshold for the traditional classification layer to generate results")
    parser.add_argument('--trad_ones', default=False,action='store_true',help="Whether to change the confidence level of all pseudo labels generated by the classification layer to 1")
    
    # Traing utils
    # city2foggy     sim10k2city    city2bdd100k
    parser.add_argument('--da_task', default='city2foggy',type=str,help="Task selection")
    parser.add_argument('--output_dir', default='logs/', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',default='', help='Pre-trained model address')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', 
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False,action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--visualize', default=False,action='store_true', help="Visualization of test results")
    parser.add_argument('--vis_train', default=False,action='store_true', help="Training process visualization")
    parser.add_argument('--warm_epochs', default=8, type=int,help="Burn-In epochs")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--debug',  default=False,action='store_true',
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true', 
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', 
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', default=False,action='store_true',
                        help="Train with mixed precision")
    return parser

def build_model_main(args):
    if args.modelname.lower() == 'dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    # Only DAB Deformable DETR
    elif args.modelname.lower() == 'dab_deformable_detr':
        model, criterion, postprocessors,postprocessors_eval = build_dab_deformable_detr(args)
    else:
        raise NotImplementedError
    return model, criterion, postprocessors,postprocessors_eval

def main(args):
  
    if args.da_task=="city2foggy":
      from engine.city2foggy.engine_city2foggy import evaluate, train_one_epoch
      args.output_dir=args.output_dir+'city2foggy/'
    elif args.da_task=="sim10k2city":
      from engine.sim10k2city.engine_sim10k2city import evaluate, train_one_epoch
      args.output_dir=args.output_dir+'sim10k2city/'
    elif args.da_task=="city2bdd100k":
      from engine.city2bdd100k.engine_city2bdd100k import evaluate, train_one_epoch
      args.output_dir=args.output_dir+'city2bdd100k/'
      
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    current_time = now.strftime('%Y%m%d%H%M')
    if args.eval:
      args.output_dir=args.output_dir+'eval_'+current_time
    else:
      args.output_dir=args.output_dir+'train_'+current_time
    best_performance=0
    utils.init_distributed_mode(args)
    # print
    print("Current task is:", args.da_task)
    print("Learning rate is:", args.lr)
    print("Pseudo-label confidence for classification layer is:", args.pre_conf)
    print("Whether to enable traditional scores all set to 1:", args.trad_ones)
    print("Whether to enable class-wise scale clustering:", args.open_pt)
    print("low_conf is:", args.low_conf)
    print("high_conf is:", args.high_conf)
    print("Cosine similarity threshold is:", args.cos_thre)
    print("Learning prompt score threshold is:", args.learn_thre)
    print("Number of TOP selections is:", args.top_num)
    print("Low confidence IOU is:", args.low_iou)
    print("High confidence IOU is:", args.high_iou)
    print("Final IOU is:", args.last_iou)
    print("Model to resume training is:", args.resume)
    print("Pretrained model path is:", args.pretrain_model_path)
    # wandb
    config = dict (
      batch_size=args.batch_size,
      epochs = args.epochs,
      lr = args.lr,
      lr_drop=args.lr_drop,
      con_weight=args.con_weight,
      dis_weight=args.dis_weight,
      pre_conf=args.pre_conf,
      weight_decay = args.weight_decay,
      pretrain_model_path=args.pretrain_model_path,
      warm_epochs=args.warm_epochs,
      open_pt=args.open_pt,
      low_conf=args.low_conf,
      high_conf=args.high_conf,
      top_num=args.top_num,
      low_iou=args.low_iou,
      high_iou=args.high_iou,
      last_iou=args.last_iou,
      dataset = args.coco_path,
      resume=args.resume,
      model_ema_decay=args.model_ema_decay,
      cos_thre=args.cos_thre,
      da_task=args.da_task
    )
    my_wandb=None
    if not args.eval and not args.debug:
      my_wandb=wandb.init(
        project="DPGT",
        config=config,
      )
    if args.frozen_weights is not None:
      assert args.masks, "Frozen training is meant for segmentation only"
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="DAB-DETR")
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    model, criterion, postprocessors,postprocessors_eval = build_model_main(args)
    wo_class_error = False
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    dataset_train_s = build_dataset(image_set='train', args=args,domain1='source_r',domain2='source_f')
    dataset_train_t = build_dataset(image_set='train', args=args,domain1='target_r',domain2='target_f')
    dataset_val = build_dataset(image_set='val', args=args,domain1='target_r',domain2='target_f')

    if args.distributed:
        sampler_train_s = DistributedSampler(dataset_train_s)
        sampler_train_t = DistributedSampler(dataset_train_t)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train_s = torch.utils.data.SequentialSampler(dataset_train_s)
        sampler_train_t = torch.utils.data.RandomSampler(dataset_train_t)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_s, args.batch_size, drop_last=True)
    batch_sampler_train_t = torch.utils.data.BatchSampler(sampler_train_t, args.batch_size, drop_last=True)

    data_loader_train_s = DataLoader(dataset_train_s, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_t = DataLoader(dataset_train_t, batch_sampler=batch_sampler_train_t,collate_fn=utils.collate_fn, num_workers=args.num_workers)
  
    data_loader_val = DataLoader(dataset_val,3*args.batch_size, sampler=sampler_val,drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # Iterate over the model parameters in the checkpoint
            if 'tea' in args.resume:
              new_state_dict = {}
              for key, value in checkpoint['model'].items():
                  # Drop the 'module.' prefix
                  new_key = key.replace('module.', '')
                  new_state_dict[new_key] = value
              checkpoint['model']=new_state_dict
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("The current epoch is:"+str(args.start_epoch))
        if not args.eval and not args.debug:
          test_stats, coco_evaluator = evaluate(
              model, criterion, postprocessors_eval, data_loader_val, base_ds, device, args.output_dir,
              wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
          )
          best_performance = test_stats['coco_eval_bbox'][1]
          print("The AP50 for the current resume model is:"+str( round(test_stats['coco_eval_bbox'][1] * 100, 2))+'%')

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        model_dict=model_without_ddp.state_dict()
        if 'tea' in args.pretrain_model_path:
              print("Teacher models pre-trained using source domains")
              new_state_dict = {}
              for key, value in checkpoint.items():
                  # Drop the 'module.' prefix
                  new_key = key.replace('module.', '')
                  new_state_dict[new_key] = value
              checkpoint=new_state_dict
        #if coco model
        if checkpoint['class_embed.5.weight'].shape[0]==91:
              print("Pre-train models with coco")
              for i in range(6):
                i_weight='class_embed.'+str(i)+'.weight'
                i_bias='class_embed.'+str(i)+'.bias'
                if args.da_task=="city2foggy":
                  class_i_weight = torch.stack([
                    checkpoint[i_weight][0],
                    checkpoint[i_weight][3],
                    checkpoint[i_weight][1],
                    checkpoint[i_weight][1],
                    checkpoint[i_weight][2],
                    checkpoint[i_weight][4],
                    checkpoint[i_weight][6],
                    checkpoint[i_weight][8],
                    checkpoint[i_weight][7]
                    ])
                  class_i_bias = torch.stack([
                    checkpoint[i_bias][0],
                    checkpoint[i_bias][3],
                    checkpoint[i_bias][1],
                    checkpoint[i_bias][1],
                    checkpoint[i_bias][2],
                    checkpoint[i_bias][4],
                    checkpoint[i_bias][6],
                    checkpoint[i_bias][8],
                    checkpoint[i_bias][7]
                    ])
                elif args.da_task=="sim10k2city":
                  class_i_weight = torch.stack([
                    checkpoint[i_weight][0],
                    checkpoint[i_weight][3],
                    ])
                  class_i_bias = torch.stack([
                    checkpoint[i_bias][0],
                    checkpoint[i_bias][3],
                    ])
                elif args.da_task=="city2bdd100k":
                  class_i_weight = torch.stack([
                    checkpoint[i_weight][0],
                    checkpoint[i_weight][1],
                    checkpoint[i_weight][3],
                    checkpoint[i_weight][6],
                    checkpoint[i_weight][1],
                    checkpoint[i_weight][8],
                    checkpoint[i_weight][4],
                    checkpoint[i_weight][2]
                    ])
                  class_i_bias = torch.stack([
                    checkpoint[i_bias][0],
                    checkpoint[i_bias][1],
                    checkpoint[i_bias][3],
                    checkpoint[i_bias][6],
                    checkpoint[i_bias][1],
                    checkpoint[i_bias][8],
                    checkpoint[i_bias][4],
                    checkpoint[i_bias][2]
                    ])
                checkpoint[i_weight]=class_i_weight
                checkpoint[i_bias]=class_i_bias
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
   
    if args.model_ema:
        model_t = ModelEma(model,decay=args.model_ema_decay,device=device)
        #Note that model_t.module is the model
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors_eval,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        print("The tested AP50 is: " + str(round(test_stats['coco_eval_bbox'][1] * 100, 2)) + '%')
        print("The tested AR50 is: " + str(round(test_stats['coco_eval_bbox'][8] * 100, 2)) + '%')
        print("The tested loss is: " + str(round(test_stats['loss'], 2)))
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    # time.sleep(5)
    args.visualize=False
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Using the best teacher models for the burn-in phase
        if epoch==args.warm_epochs:
          checkpoint = torch.load(output_dir / f'checkpoint_best_tea.pth', map_location='cpu')['model']
          print("Using the best teacher models for the burn-in phase")
          new_state_dict = {}
          for key, value in checkpoint.items():
              # Drop the 'module.' prefix
              new_key = key.replace('module.', '')
              new_state_dict[new_key] = value
          checkpoint=new_state_dict
          _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
          from collections import OrderedDict
          def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True
          _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
          _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
          model_t = ModelEma(model,decay=args.model_ema_decay,device=device)

        
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train_s.set_epoch(epoch)
            sampler_train_t.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, model_t,criterion, postprocessors,data_loader_train_s,data_loader_train_t,optimizer, device, epoch,args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None),my_wandb=my_wandb)
        #lr step
        lr_scheduler.step()
        #save model
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        # #eval teacher
        test_stats, coco_evaluator = evaluate(
            model_t.module, criterion, postprocessors_eval, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        print("The teacher model AP50 is:"+str( round(test_stats['coco_eval_bbox'][1] * 100, 2))+'%')
        print("The teacher model AR50 is:"+str( round(test_stats['coco_eval_bbox'][8] * 100, 2))+'%')
        print("The LOSS for the teacher modeling test is:"+str( round(test_stats['loss'], 2)))
        #find best model teacher
        targetAP50_tea = test_stats['coco_eval_bbox'][1]
        if targetAP50_tea > best_performance:
            best_performance = targetAP50_tea
            print("There are better metrics out there in the teacher model")
            checkpoint_paths.append(output_dir / f'checkpoint_best_tea.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_t.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        if not args.debug:
          my_wandb.log({
                    "tea_test_loss":round(test_stats['loss'], 2),
                    "tea_bbox AP": round(test_stats['coco_eval_bbox'][1] * 100, 2),
                    "tea_bbox AR": round(test_stats['coco_eval_bbox'][8] * 100, 2),
                    })
                
        #eval student
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors_eval, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        print("The student model AP50 is:"+str( round(test_stats['coco_eval_bbox'][1] * 100, 2))+'%')
        print("The student model AR50 is:"+str( round(test_stats['coco_eval_bbox'][8] * 100, 2))+'%')
        print("The LOSS for the student model test is:"+str( round(test_stats['loss'], 2)))
        #find best model student
        targetAP50 = test_stats['coco_eval_bbox'][1]
        if targetAP50 > best_performance:
            best_performance = targetAP50
            print("There are better metrics out there in the student model")
            checkpoint_paths.append(output_dir / f'checkpoint_best_stu.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        #log
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if not args.debug:
          my_wandb.log({
                    "test_loss":round(test_stats['loss'], 2),
                    "bbox AP": round(test_stats['coco_eval_bbox'][1] * 100, 2),
                    "bbox AR": round(test_stats['coco_eval_bbox'][8] * 100, 2),
                    })
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("The best mAP50 metrics are:"+str(round(best_performance * 100, 2))+'%')
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))
    if not args.debug:
      wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
