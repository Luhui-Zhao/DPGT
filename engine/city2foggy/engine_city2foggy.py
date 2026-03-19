# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import gc 
import math
import os
import sys
from typing import Iterable
from util.utils import to_device
import torch.nn.functional as F
import torch
import copy
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import numpy as np
import cv2
from torchvision.ops import nms
import pandas as pd
import torch.distributed as dist
import pickle
import time
from torchvision.ops import box_iou
# target
Prompt_vectors_t = [[] for _ in range(8)]
area_ratios_t = [[] for _ in range(8)]

# target false
Prompt_vectors_tf = [[] for _ in range(8)]
area_ratios_tf = [[] for _ in range(8)]
debug_id=0
num_class=8
num_scale=25
# Training
def train_one_epoch(model: torch.nn.Module,model_t: torch.nn.Module, criterion: torch.nn.Module,postprocessors,data_loader_s: Iterable,data_loader_t: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,my_wandb=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    need_tgt_for_training= False
    model.train()
    criterion.train()
    model_t.module.eval()
    metric_logger = utils.MetricLogger(delimiter="  ",my_wandb=my_wandb)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print("Now for a new epoch:",epoch)
    # Print frequency
    print_freq = 100
    train_state=True
    # Check whether it is in the burn-in phase
    burn_in=True
    args.learn_thre=args.pre_conf
    if epoch>=args.warm_epochs:
      print("The target field is currently introduced")
      burn_in=False
      args.learn_thre=0.5
    print("The threshold for learning is:",args.learn_thre)
    print("burn_in?:"+str(burn_in))
    # Create the file path of the cluster prompt
    prompt_path=os.path.join(args.output_dir, f'prompt')
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(prompt_path+"/prompts_avg", exist_ok=True)
    prompt_pth_tf_dir=prompt_path+"/prompts_avg/tf_"+str(epoch-1)+".pth"
    prompt_pth_t_dir=prompt_path+"/prompts_avg/t_"+str(epoch-1)+".pth"
    # Read local clustering hints (from the previous round)
    prompt_blank=True
    if not os.path.exists(prompt_pth_t_dir):
      query_t_last= [[[] for _ in range(num_scale)] for _ in range(num_class)]
      print("No local target domain clustering features")
    else:
      query_t_last = torch.load(prompt_pth_t_dir)
      prompt_blank=False
      print("Read local clustering features:",prompt_pth_t_dir)
    if not os.path.exists(prompt_pth_tf_dir):
      query_tf_last= [[[] for _ in range(num_scale)] for _ in range(num_class)]
      print("No local pseudo-target domain clustering features")
    else:
      query_tf_last = torch.load(prompt_pth_tf_dir)
      prompt_blank=False
      print("Read local clustering features:",prompt_pth_tf_dir)
    # Change the clustering prompt from a list to a matrix vector
    if not prompt_blank:
      query_tf_tensor = torch.zeros((len(query_tf_last) * len(query_tf_last[0]), 256))
      for i in range(len(query_tf_last)):
        for j in range(len(query_tf_last[i])):
          query_tf_tensor[i * len(query_tf_last[i]) + j] = query_tf_last[i][j]
      query_t_tensor = torch.zeros((len(query_t_last) * len(query_t_last[0]), 256))
      for i in range(len(query_t_last)):
        for j in range(len(query_t_last[i])):
          query_t_tensor[i * len(query_t_last[i]) + j] = query_t_last[i][j]
    else:
      query_tf_tensor = torch.zeros((len(query_tf_last) * len(query_tf_last[0]), 256))
      query_t_tensor = torch.zeros((len(query_t_last) * len(query_t_last[0]), 256))
    query_tf_last=copy.deepcopy(query_tf_tensor)
    query_t_last=copy.deepcopy(query_t_tensor)
    query_tf_last=query_tf_last.to(device)
    query_t_last=query_t_last.to(device)
    del query_tf_tensor,query_t_tensor
    global Prompt_vectors_t
    global Prompt_vectors_tf
    # Image id when training visualization
    global img_id
    img_id=0
    _cnt = 0
    # Obtain the id of the GPU during multi-card training
    gpu_id = get_gpu_id()
    global area_ratios_t
    global area_ratios_tf
    # Read the gpu that belongs to gpu_id
    ar_dir_t=prompt_path+"/area_ratios_t_"+str(gpu_id)+".pkl"
    ar_dir_tf=prompt_path+"/area_ratios_tf_"+str(gpu_id)+".pkl"
    ar_dir_t_epo=prompt_path+"/area_ratios_t_epo_"+str(gpu_id)+".pkl"
    ar_dir_tf_epo=prompt_path+"/area_ratios_tf_epo_"+str(gpu_id)+".pkl"
    if epoch>0:
      area_ratios_t=load_prompts_from_file(ar_dir_t)
      area_ratios_tf=load_prompts_from_file(ar_dir_tf)
    # Read the gpu that belongs to gpu_id
    prompt_dir_t=prompt_path+"/prompts_t_"+str(gpu_id)+".pkl"
    prompt_dir_tf=prompt_path+"/prompts_tf_"+str(gpu_id)+".pkl"
    if epoch>0:
      Prompt_vectors_t=load_prompts_from_file(prompt_dir_t)
      Prompt_vectors_tf=load_prompts_from_file(prompt_dir_tf)

    for data_s_sf,data_t_tf in metric_logger.log_every(data_loader_s,data_loader_t, print_freq, header, logger=logger,train_state=train_state,if_wandb=args.debug,task="c2f"):

        samples_s=data_s_sf[0].to(device)#Source domain image
        targets_s=data_s_sf[1]#Source domain label
        samples_sf=data_s_sf[2].to(device)#False source domain image
        targets_sf=data_s_sf[3]#False source domain label
        samples_t=data_t_tf[0].to(device)#Target domain image
        targets_t=data_t_tf[1]#Target field label: Only image size is used
        samples_tf=data_t_tf[2].to(device)#False target domain image
        targets_tf=data_t_tf[3]#False target field label: Only image size is used
        for i in range(args.batch_size):
          targets_t[i]['scores'] = torch.ones_like(targets_t[i]['labels'])
          targets_tf[i]['scores']=torch.ones_like(targets_tf[i]['labels'])
        #Label moves to GPU
        targets_s = [{k: v.to(device) for k, v in t.items()} for t in targets_s]
        targets_sf = [{k: v.to(device) for k, v in t.items()} for t in targets_sf]
        targets_t = [{k: v.to(device) for k, v in t.items()} for t in targets_t]
        targets_tf = [{k: v.to(device) for k, v in t.items()} for t in targets_t]
        #Get image size
        target_sizes = torch.stack([t["size"] for t in targets_t], dim=0).to(device)
        ori_target_sizes = torch.stack([t["orig_size"] for t in targets_t], dim=0).to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                print("need_tgt_for_training")
            else:
                #SR-Student training
                outputs_s = model(samples_s)
                #SF-student training
                outputs_sf = model(samples_sf)
                #TR-Teacher training
                outputs_t = model(samples_t)
                #TF-teacher inference
                infer_tf = model_t.module(samples_tf)
                #TR-teacher inference
                infer_t = model_t.module(samples_t)

         #Prompt Tuning+pseudo labels tf
            # postprocessors
            results_tf = postprocessors['bbox'](infer_tf, target_sizes)
            # query by decoder  bs,100,256
            hs_query_tf=infer_tf['hs_query'].clone()
            # Get information about the query
            filt_results_tf={}
            filt_results_tf = [[(v1, v2, v3, box, query) for v1, v2, v3, box, query in [(results_tf[j]['scores'][i].item(), results_tf[j]['labels'][i].item(), results_tf[j]['id_query'][i].item(), bbox_cxcywh(results_tf[j]['boxes'], targets_tf[j]['size'])[i].unsqueeze(0), hs_query_tf[j][results_tf[j]['id_query'][i].item()]) for i in range(100)] if v1 >= args.low_conf and v1 < args.high_conf] for j in range(len(results_tf))]
            results_tf_c=copy.deepcopy(results_tf)
            # The traditional classification layer obtains confidence
            results_tf_tra,mark_compu_tf=pseudo_convert(results_tf_c,target_sizes,ori_target_sizes,args.pre_conf,args.trad_ones)#pseudo convert
            results_tf_tra = [{k: v.to(device) for k, v in t.items()} for t in results_tf_tra]
            # Prompt generation pseudo labels
            if not burn_in:
              result_tf_pt,result_tf_pt_fn=Prompt_Tuning_pseudo(filt_results_tf,hs_query_tf,query_tf_last,target_sizes,ori_target_sizes,args.cos_thre,is_t=False)
              for item in result_tf_pt:
                if item!=None:
                  item['scores'] = item['scores'].fill_(1.0)
            else:
              result_tf_pt={}
              result_tf_pt = [None for _ in range(args.batch_size)]
              result_tf_pt_fn={}
              result_tf_pt_fn = [None for _ in range(args.batch_size)]
            # The ultimate in pseudo-labeling
            if not burn_in :
              result_tf_mer=result_tf_pt
            else:
              result_tf_mer=results_tf_tra
            # t_tf_gt: pseudo labels after high confidence fusion
            t_tf_gt=copy.deepcopy(result_tf_mer)
            # t_tf_fn: pseudo labels after low confidence fusion
            t_tf_fn=copy.deepcopy(result_tf_pt_fn)
            # t_tf_fn_vis: Low-confidence fused pseudo-labels for visualization
            t_tf_fn_vis=copy.deepcopy(result_tf_pt_fn)
            # t_tf_label: final pseudo labels
            t_tf_label=copy.deepcopy(result_tf_mer)
            # Select the query for learning
            learn_tf=copy.deepcopy(results_tf_tra)
            if args.top_num!=0:
              learn_tf=learn_tops(learn_tf,args.learn_thre,args.top_num)
            # tf Prompt Tuning Update
            if epoch>=args.warm_epochs-2:
              calculate_area_ratios(learn_tf,epoch,is_t=False)
              Prompt_Tuning_tf(hs_query_tf,learn_tf,epoch,args.learn_thre)
            
         #Prompt Tuning+pseudo labels tr
            #postprocessors
            results_t = postprocessors['bbox'](infer_t, target_sizes)
            #query by decoder
            hs_query_t=infer_t['hs_query'].clone()
            # Get information about the query
            results_t_cxcywh=filt_results_t={}
            filt_results_t = [[(v1, v2, v3, box, query) for v1, v2, v3, box, query in [(results_t[j]['scores'][i].item(), results_t[j]['labels'][i].item(), results_t[j]['id_query'][i].item(), bbox_cxcywh(results_t[j]['boxes'], targets_t[j]['size'])[i].unsqueeze(0), hs_query_t[j][results_t[j]['id_query'][i].item()]) for i in range(100)] if v1 >= args.low_conf and v1 < args.high_conf] for j in range(len(results_t))]
            results_t_c=copy.deepcopy(results_t)
            # The traditional classification layer obtains confidence
            results_t_tra,mark_compu_t=pseudo_convert(results_t_c,target_sizes,ori_target_sizes,args.pre_conf,args.trad_ones)
            results_t_tra = [{k: v.to(device) for k, v in t.items()} for t in results_t_tra]
            # Prompt generation pseudo labels
            if not burn_in:
              result_t_pt,result_t_pt_fn=Prompt_Tuning_pseudo(filt_results_t,hs_query_t,query_t_last,target_sizes,ori_target_sizes,args.cos_thre,is_t=True)
            else:
              result_t_pt={}
              result_t_pt = [None for _ in range(args.batch_size)]
              result_t_pt_fn={}
              result_t_pt_fn = [None for _ in range(args.batch_size)]
            # The ultimate in pseudo-labeling
            if not burn_in :
              result_t_mer=result_t_pt
            else:
              result_t_mer=results_t_tra
            # Select the query for learning
            learn_t=copy.deepcopy(results_t_tra)
            if args.top_num!=0:
              learn_t=learn_tops(learn_t,args.learn_thre,args.top_num)
            # tr Prompt Tuning Update
            if epoch>=args.warm_epochs-2:
              calculate_area_ratios(learn_t,epoch,is_t=True)
              Prompt_Tuning_t(hs_query_t,learn_t,epoch,args.learn_thre)
            
            # For low confidence
            for i in range(len(result_t_mer)):
              if result_t_pt_fn[i]!=None and result_tf_pt_fn[i]!=None:
                # Low confidence fusion of two domains
                t_tf_fn[i]=merge_dicts(result_t_pt_fn[i], result_tf_pt_fn[i])
                # Overlap-Selective Suppression
                t_tf_fn[i]['scores'], t_tf_fn[i]['labels'], t_tf_fn[i]['boxes'], mask = custom_nms(scores=t_tf_fn[i]['scores'],labels=t_tf_fn[i]['labels'],boxes=t_tf_fn[i]['boxes'],image_size=t_tf_fn[i]['size'][:2],iou_threshold=args.low_iou,device=device)
                t_tf_fn[i]['cxcywh']=t_tf_fn[i]['cxcywh'][mask]
                t_tf_fn[i]['id_query']=t_tf_fn[i]['id_query'][mask]
                t_tf_fn[i]['size']=t_tf_fn[i]['size'][:2]
              else:
                t_tf_fn[i]=None
            
            # For high confidence and final result
              high_pt = []

              for i in range(len(result_t_mer)):

                  t_res = result_t_mer[i]
                  tf_res = result_tf_mer[i]

                  if t_res is None and tf_res is None:
                      t_tf_gt[i] = {}
                      t_tf_gt[i]['boxes'] = torch.empty((0, 4)).to(device)
                      t_tf_gt[i]['labels'] = torch.empty(0, dtype=torch.int64).to(device)
                      high_pt.append(None)
                      continue

                  if tf_res is None:
                      t_tf_gt[i] = t_res
                      high_pt.append(t_tf_gt[i])
                      continue

                  if t_res is None:
                      t_tf_gt[i] = tf_res
                      high_pt.append(t_tf_gt[i])
                      continue

                  filtered_t = remove_overlapping_boxes(
                      t_res,
                      tf_res,
                      iou_threshold=args.high_iou,
                      device=device
                  )

                  t_tf_gt[i] = merge_dicts(filtered_t, tf_res)

                  high_pt.append(t_tf_gt[i])
            
            # All pseudo labels are on the NMS together
            for i in range(len(result_t_mer)):
              if t_tf_fn[i] != None and t_tf_fn[i]['labels'].shape[0] != 0 and t_tf_gt[i]['labels'].shape[0]!=0:
                t_tf_fn_vis = copy.deepcopy(t_tf_fn)
                keep_indices = []
                for m, (fn_box, fn_label) in enumerate(zip(t_tf_fn[i]['cxcywh'], t_tf_fn[i]['labels'])):
                    iou_max = 0
                    for gt_box, gt_label in zip(t_tf_gt[i]['cxcywh'], t_tf_gt[i]['labels']):
                        if fn_label == gt_label:
                            iou = box_iou_cxcywh(fn_box.unsqueeze(0), gt_box.unsqueeze(0))
                            if iou > iou_max:
                                iou_max = iou
                    keep_indices.append(iou_max <= args.last_iou)
                keep_indices = torch.tensor(keep_indices, device=t_tf_fn[i]['boxes'].device)
                for key in ['boxes', 'cxcywh', 'labels', 'scores', 'id_query']:
                    t_tf_fn[i][key] = t_tf_fn[i][key][keep_indices]
              else:
                  t_tf_fn_vis[i] = t_tf_fn[i]=None
            
            # merge all labels
            for i in range(len(result_t_mer)):
              t_tf_label[i]=merge_dicts(t_tf_gt[i], t_tf_fn[i])
            t_tf_label = process_tf_labels(t_tf_label)      
   
            #source loss
            loss_dict_s = criterion(outputs_s, targets_s)#sr loss
            loss_dict_sf = criterion(outputs_sf, targets_sf)#sf loss
            #consistency loss
            loss_dict_con = {key:torch.abs( loss_dict_s[key] - loss_dict_sf[key])**2*args.con_weight for key in loss_dict_s}
            #distil loss
            loss_dict_t = criterion(outputs_t, t_tf_label)#dis loss
            loss_dict_dis = {key: value * args.dis_weight for key, value in loss_dict_t.items()}
            #total loss
            weight_dict = criterion.weight_dict
            if burn_in:
              loss_dict = {key: loss_dict_s[key] + loss_dict_sf[key] +loss_dict_con[key] for key in loss_dict_s}
            else:
              loss_dict = {key: loss_dict_s[key] + loss_dict_sf[key] +loss_dict_con[key]+loss_dict_dis[key] for key in loss_dict_s}
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses.backward()
            model_t.update(model)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        del samples_s, samples_sf, samples_t, samples_tf, targets_s, targets_sf, targets_t, targets_tf
        del outputs_s, outputs_sf, outputs_t, infer_tf, infer_t, results_tf, results_tf_c, results_tf_tra
        del results_t, results_t_c, results_t_tra, t_tf_gt, loss_dict_s, loss_dict_sf, loss_dict_con, loss_dict_t, loss_dict_dis
        del losses, loss_dict, loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled
        _cnt += 1

    # Proportion statistics
    # Proportion statistics are saved locally
    save_file_to_pkl(area_ratios_t, ar_dir_t)
    save_file_to_pkl(area_ratios_tf, ar_dir_tf)
    area_ratios_t_epoch = []
    for i in range(len(area_ratios_t)):
      area_ratios_t_epoch.append([sublist[0] for sublist in area_ratios_t[i] if sublist[1] in {epoch}])
    area_ratios_tf_epoch = []
    for i in range(len(area_ratios_tf)):
      area_ratios_tf_epoch.append([sublist[0] for sublist in area_ratios_tf[i] if sublist[1] in {epoch}])
    save_file_to_pkl(area_ratios_t_epoch, ar_dir_t_epo)
    save_file_to_pkl(area_ratios_tf_epoch, ar_dir_tf_epo)
    # Read the Proportion statistics address
    file_list_t = []
    file_list_tf = []
    file_list_t_epo = []
    file_list_tf_epo = []
    for w in range(args.world_size):
      filename_t=prompt_path+"/area_ratios_t_"+str(w)+".pkl"
      filename_tf=prompt_path+"/area_ratios_tf_"+str(w)+".pkl"
      filename_t_epo=prompt_path+"/area_ratios_t_epo_"+str(w)+".pkl"
      filename_tf_epo=prompt_path+"/area_ratios_tf_epo_"+str(w)+".pkl"
      file_list_t.append(filename_t)
      file_list_tf.append(filename_tf)
      file_list_t_epo.append(filename_t_epo)
      file_list_tf_epo.append(filename_tf_epo)
    time.sleep(0.2)
    # Read the Proportion statistics and merge they
    area_ratios_t = merge_ar_files(file_list_t_epo)
    area_ratios_tf = merge_ar_files(file_list_tf_epo)
    inter_t=interval_seg(area_ratios_t,is_t=True)
    inter_tf=interval_seg(area_ratios_tf,is_t=False)
    
    # Features are saved locally
    save_file_to_pkl(Prompt_vectors_t, prompt_dir_t)
    save_file_to_pkl(Prompt_vectors_tf, prompt_dir_tf)
    # Read the feature address
    file_list_t = []
    file_list_tf = []
    for w in range(args.world_size):
      filename_t=prompt_path+"/prompts_t_"+str(w)+".pkl"
      filename_tf=prompt_path+"/prompts_tf_"+str(w)+".pkl"
      file_list_t.append(filename_t)
      file_list_tf.append(filename_tf)
    time.sleep(0.2)
    # Read the prompt and merge prompt
    Prompt_vectors_t = merge_prompts_files(file_list_t)
    Prompt_vectors_tf = merge_prompts_files(file_list_tf)
    # Average and save
    target_device = torch.device('cpu')
    # Read the prompt address
    prompt_pth_tf_dir=prompt_path+"/prompts_avg/tf_"+str(epoch)+".pth"
    prompt_pth_t_dir=prompt_path+"/prompts_avg/t_"+str(epoch)+".pth"
    #  Assign features to intervals
    if inter_tf!=[] and inter_t!=[]:
      Prompt_vectors_tf=interval_all(Prompt_vectors_tf,inter_tf,burn_in,epoch)
      Prompt_vectors_t=interval_all(Prompt_vectors_t,inter_t,burn_in,epoch)
      #Average and save
      total_tf=prompt_save(Prompt_vectors_tf,target_device,prompt_pth_tf_dir)
      total_t=prompt_save(Prompt_vectors_t,target_device,prompt_pth_t_dir)
      if not args.debug:
        my_wandb.log({
                        "total_tf":total_tf,
                        "total_t": total_t,
                        })
      # table display
      tab_display(Prompt_vectors_tf,Prompt_vectors_t,total_tf,total_t)
    # # Read the gpu that belongs to gpu_i

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    torch.cuda.empty_cache()  # Clear cache
    gc.collect()  # Operational garbage collection
    torch.cuda.synchronize()  # Synchronous GPU operations
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None,my_wandb=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ",my_wandb=my_wandb)
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    train_state=False
    data_loader_sf=None
    for data_val in metric_logger.log_every(data_loader, data_loader_sf,100, header, logger=logger,train_state=train_state,if_wandb=args.debug):
        
        #for target real
        samples_r = data_val[0][0].to(device)
        targets_r = [{k: to_device(v, device) for k, v in t.items()} for t in data_val[0][1]]
        for i in range(len(data_val[0][1])):
          data_val[0][1][i]['scores']=torch.ones_like(data_val[0][1][i]['labels'])
        
         #for target false 
        samples_f = data_val[0][2].to(device)
        targets_f = [{k: to_device(v, device) for k, v in t.items()} for t in data_val[0][3]]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples_r, targets_r)
            else:
                outputs_r = model(samples_r)
                outputs_f = model(samples_f)
            # outputs = model(samples)

            loss_dict = criterion(outputs_r, targets_r)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets_r], dim=0)
        results_r = postprocessors['bbox'](outputs_r, orig_target_sizes)
        results = copy.deepcopy(results_r)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets_r], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets_r, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets_r):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator

# Traditional pseudo label acquisition
def pseudo_convert(result,img_size,ori_img_size,conf,trad_ones):
  result_infer=copy.deepcopy(result)
  mark_compu = torch.ones(len(result_infer))
  def convert_box(boxes,img_size):
      x1 = boxes[:, 0]
      y1 = boxes[:, 1]
      x2 = boxes[:, 2]
      y2 = boxes[:, 3]
      # Central coordinate
      cx = (x1 + x2) / 2
      cy = (y1 + y2) / 2
      # Width and height
      w = x2 - x1
      h = y2 - y1
      # Create a new tensor containing cx, cy, w, h
      cxcywh_tensor = torch.stack((cx, cy, w, h), dim=1)
      # normalization
      normalized_cxcywh = cxcywh_tensor.clone()
      normalized_cxcywh[:, 0] /= img_size[i][1]   # cx normalization
      normalized_cxcywh[:, 1] /= img_size[i][0]   # cy normalization
      normalized_cxcywh[:, 2] /= img_size[i][1]   # w normalization
      normalized_cxcywh[:, 3] /= img_size[i][0]  # h normalization
      return normalized_cxcywh,cxcywh_tensor
  for i in range(len(result_infer)):
    # Source data processing
    indices = torch.nonzero(result_infer[i]['scores'] > conf).squeeze()
    if indices.numel()==0:
      mark_compu[i]=0
      # return last one
      last_index=1
      result_infer[i]['scores']=result_infer[i]['scores'][:last_index]
      result_infer[i]['id_query']=result_infer[i]['id_query'][:last_index]
      result_infer[i]['labels']=result_infer[i]['labels'][:last_index]
      result_infer[i]['boxes']=result_infer[i]['boxes'][:last_index,]
      result_infer[i]['boxes'],result_infer[i]['cxcywh']=convert_box(result_infer[i]['boxes'],img_size)
      result_infer[i]['orig_size']=ori_img_size[i]
      result_infer[i]['size']=img_size[i]
    else:
      if indices.dim()==0:
        last_index=1
      else:
        last_index = indices[-1].item()+1
      result_infer[i]['scores']=result_infer[i]['scores'][:last_index]
      result_infer[i]['id_query']=result_infer[i]['id_query'][:last_index]
      result_infer[i]['labels']=result_infer[i]['labels'][:last_index]
      result_infer[i]['boxes']=result_infer[i]['boxes'][:last_index,]
      result_infer[i]['boxes'],result_infer[i]['cxcywh']=convert_box(result_infer[i]['boxes'],img_size)
      result_infer[i]['orig_size']=ori_img_size[i]
      result_infer[i]['size']=img_size[i]
  # The traditional method has a confidence level of 1
  if trad_ones:
    for i in range(len(result_infer)):
      result_infer[i]['scores']=torch.ones_like(result_infer[i]['scores'])
  return result_infer,mark_compu
    
def remove_overlapping_boxes(t_dict, tf_dict, iou_threshold=0.5, device=None):

    if t_dict is None:
        return None

    if tf_dict is None:
        return t_dict

    t_boxes = cxcywh_to_xyxy(t_dict['boxes'])
    tf_boxes = cxcywh_to_xyxy(tf_dict['boxes'])

    img_h, img_w = t_dict['size'][:2]
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)

    t_boxes = t_boxes * scale
    tf_boxes = tf_boxes * scale

    t_labels = t_dict['labels']
    tf_labels = tf_dict['labels']

    keep_mask = torch.ones(t_boxes.size(0), dtype=torch.bool).to(device)

    unique_labels = t_labels.unique()

    for label in unique_labels:
        t_mask = (t_labels == label)
        tf_mask = (tf_labels == label)

        if tf_mask.sum() == 0:
            continue

        t_cls_boxes = t_boxes[t_mask]
        tf_cls_boxes = tf_boxes[tf_mask]

        ious = box_iou(t_cls_boxes, tf_cls_boxes)

        max_iou, _ = ious.max(dim=1)
        remove = max_iou > iou_threshold

        keep_mask[t_mask.nonzero(as_tuple=True)[0][remove]] = False

    filtered_dict = {}

    for key in t_dict.keys():
        if isinstance(t_dict[key], torch.Tensor) and t_dict[key].shape[0] == keep_mask.shape[0]:
            filtered_dict[key] = t_dict[key][keep_mask]
        else:
            filtered_dict[key] = t_dict[key]

    return filtered_dict
    
    
# Fusing two target domain results
def merge_dicts(dict1, dict2):
    if dict1==None:
      return dict2
    elif dict2==None:
      return dict1
    merged_dict = {}
    # Process the first dictionary
    for key, value in dict1.items():
        if key in merged_dict:
            merged_dict[key] = torch.cat((merged_dict[key], value))
        else:
            merged_dict[key] = value.clone()  # Use clone() to avoid changing the original dictionary

    # Handling the second dictionary
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] = torch.cat((merged_dict[key], value))
        else:
            merged_dict[key] = value.clone()
    
    return merged_dict
  
# cxcywh to xyxy
def cxcywh_to_xyxy(boxes):
    """Convert boxes from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)"""
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# NMS for t and tf
def perform_nms_per_class(scores, labels, boxes, image_size, iou_threshold=0.5,device=None):
    """Perform Non-Maximum Suppression (NMS) on the boxes for each class separately and return a mask of kept boxes"""
    # Convert boxes from cxcywh to xyxy
    boxes_xyxy = cxcywh_to_xyxy(boxes)
    
    # Scale the boxes to the original image size
    img_h, img_w = image_size
    scale_factors = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    scale_factors=scale_factors.to(device)
    boxes_xyxy = boxes_xyxy * scale_factors
    
    # Initialize the mask to all False
    mask = torch.zeros(scores.size(0), dtype=torch.bool)
    
    # Perform NMS for each class separately
    unique_labels = labels.unique()
    for label in unique_labels:
        class_mask = (labels == label)
        class_scores = scores[class_mask]
        class_boxes = boxes_xyxy[class_mask]
        keep = nms(class_boxes, class_scores, iou_threshold)
        mask[class_mask.nonzero(as_tuple=True)[0][keep]] = True
    
    # Filter the results
    final_scores = scores[mask]
    final_labels = labels[mask]
    final_boxes = boxes[mask]
    
    return final_scores, final_labels, final_boxes, mask
  
# tf Multi-scale multi-class prompt learn
def Prompt_Tuning_tf(query,infer,epoch,learn_thre):
  for bs in range(len(infer)):
    if infer[bs]!=None:
      areas = infer[bs]['boxes'][:, 2] * infer[bs]['boxes'][:, 3]
      for index, label in enumerate(infer[bs]['labels']):
        score=round(infer[bs]['scores'][index].item(), 2)
        if score>=learn_thre:
          query_sele = query[bs][int(infer[bs]['id_query'][index])].cpu()
          Prompt_vectors_tf[int(label) - 1].append({
            'features': query_sele, 
            'area_ratio': areas[index].item(), 
            'epoch': epoch
        })

# t Multi-scale multi-class prompt learn
def Prompt_Tuning_t(query,infer,epoch,learn_thre):
  for bs in range(len(infer)):
    if infer[bs]!=None:
      areas = infer[bs]['boxes'][:, 2] * infer[bs]['boxes'][:, 3]
      for index, label in enumerate(infer[bs]['labels']):
        score=round(infer[bs]['scores'][index].item(), 2)
        if score>=learn_thre:
          query_sele = query[bs][int(infer[bs]['id_query'][index])].cpu()
          Prompt_vectors_t[int(label) - 1].append({
            'features': query_sele, 
            'area_ratio': areas[index].item(), 
            'epoch': epoch
        })
    
# xyxy to cxcywh
def bbox_cxcywh(bbox,pic_size):
  img_height = pic_size[0]
  img_width = pic_size[1]
  cxcywh_bbox = torch.zeros_like(bbox)
  cxcywh_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # cx = (x1 + x2) / 2
  cxcywh_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # cy = (y1 + y2) / 2
  cxcywh_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  # w = x2 - x1
  cxcywh_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  # h = y2 - y1
  return cxcywh_bbox

# Generate pseudo tags by prompting
def Prompt_Tuning_pseudo(filt_results,hs_query,query_last,img_size,ori_img_size,cos_thre,is_t):
  if is_t:
    cos_thre=cos_thre+0.03
  device = query_last.device
  result_pt= []
  result_pt_fn= []
  def Postprocess_cos(bboxs,labels,scores,id_querys,img_size,ori_img_size):
    result_pt= {}
    bboxs = torch.cat(bboxs, dim=0)
    labels = torch.tensor(labels)
    scores = torch.tensor(scores)
    id_querys = torch.tensor(id_querys)
    img_height, img_width = img_size[0], img_size[1]
    # normalization
    bboxs_normalized = bboxs.clone()
    bboxs_normalized[:, 0] /= img_width
    bboxs_normalized[:, 1] /= img_height
    bboxs_normalized[:, 2] /= img_width
    bboxs_normalized[:, 3] /= img_height
    result_pt['boxes']=bboxs_normalized
    result_pt['cxcywh']=bboxs
    result_pt['labels']=labels
    result_pt['scores']=scores
    result_pt['id_query']=id_querys
    result_pt['size']=img_size
    result_pt['orig_size']=ori_img_size
    return result_pt
  
  query_tensor = torch.zeros((len(filt_results) * 100, 256))
  query_tensor[:] = torch.stack([filt_results[i][j][4] for i in range(len(filt_results)) for j in range(100)])
  query_tensor=query_tensor.to(device)
  qt_norm = torch.norm(query_tensor, dim=1, keepdim=True)
  ql_norm = torch.norm(query_last, dim=1, keepdim=True)
  # Calculate the dot product of two matrices
  dot_product = torch.mm(query_tensor, query_last.t())
  # Calculate the cosine similarity
  similarity = dot_product / (qt_norm * ql_norm.t())
  # Replace the NaN value with -inf
  similarity[similarity != similarity] = float('-inf')
  # Find the maximum value and index of each line of similarity
  max_values, max_indices = torch.max(similarity, dim=1)
  max_labels = torch.div(max_indices, num_scale, rounding_mode='floor')
  cos_thre_list=[cos_thre,cos_thre,cos_thre,cos_thre,cos_thre,cos_thre,cos_thre,cos_thre]
  for n in range(len(filt_results)):
    maxi_list = [[j, int(max_labels[100*n+j] + 1), float(max_values[100*n+j]), int(filt_results[n][j][2])] for j in range(100)]
    cosine_similarity_list = [[j, int(max_labels[100*n+j] + 1), float(max_values[100*n+j]), int(filt_results[n][j][2])] for j in range(100) if max_values[100*n+j] >= cos_thre_list[int(max_labels[100*n+j])]]
    bboxs=[]
    labels=[]
    scores=[]
    id_querys=[]
    # The similarity of all queries did not exceed the threshold
    if len(cosine_similarity_list)==0:
      result_pt.append({})
      result_pt[n]=None
    else:
      # There are queries whose similarity exceeds the threshold
      for i, k,j,m in cosine_similarity_list:
          bboxs.append(filt_results[n][i][3])
          labels.append(k)
          scores.append(j)
          id_querys.append(m)
      result_pt.append({})
      result_pt[n]=Postprocess_cos(bboxs,labels,scores,id_querys,img_size[n],ori_img_size[n])
    filtered_list = [(i, val[2]) for i, val in enumerate(maxi_list) if val[2] < cos_thre]
    sorted_filtered_list = sorted(filtered_list, key=lambda x: x[1], reverse=True)
    top_three_indices = [item[0] for item in sorted_filtered_list[:10]]
    FN_list= []
    for i in range(len(top_three_indices)):
      FN_list.append(maxi_list[top_three_indices[i]])
    bboxs_fn=[]
    labels_fn=[]
    scores_fn=[]
    id_querys_fn=[]
    for i, k,j,m in FN_list:
      if j>=cos_thre-0.05:
          bboxs_fn.append(filt_results[n][i][3])
          labels_fn.append(k)
          scores_fn.append(j)
          id_querys_fn.append(m)
    if len(labels_fn)!=0:
      result_pt_fn.append({})
      result_pt_fn[n]=Postprocess_cos(bboxs_fn,labels_fn,scores_fn,id_querys_fn,img_size[n],ori_img_size[n])
    else:
      result_pt_fn.append({})
      result_pt_fn[n]=None
    if result_pt[n]!=None:
      result_pt[n]=delete_repetion(result_pt[n])
    if result_pt_fn[n]!=None:
      result_pt_fn[n]=delete_repetion(result_pt_fn[n])
  # Moving to GPU
  for item in result_pt:
    if item!=None:
      for key, value in item.items():
          if isinstance(value, torch.Tensor):
              item[key] = value.to(device)
  for item in result_pt_fn:
    if item!=None:
      for key, value in item.items():
          if isinstance(value, torch.Tensor):
              item[key] = value.to(device)
  return result_pt,result_pt_fn

# Fusion of traditional results and prompt results
def merge_lists(list1, list2):
    merged_list = []
    def merge_dicts(dict1, dict2):
      merged_dict = {}
      for key in dict1.keys():
          if key in dict2:
              merged_dict[key] = torch.cat((dict1[key], dict2[key]))
          else:
              merged_dict[key] = dict1[key]

      for key in dict2.keys():
          if key not in dict1:
              merged_dict[key] = dict2[key]

      return merged_dict
    if list2[0]==None:
      return list1
    for d1, d2 in zip(list1, list2):
        merged_list.append(merge_dicts(d1, d2))
    return merged_list

# Displays the number of feature queries at different scales
def tab_display(Prompt_vectors_tf,Prompt_vectors_t,total_tf,total_t):
  category_instancesonly = ['car','person','rider','bicycle','motorcycle','bus','truck','train']

  scales = ['n', 's', 'm', 'l', 'X']
  k = num_scale // 5
  
  index_mapping = {}
  idx = 1
  
  for scale in scales:
      for i in range(1, k + 1):
          index_mapping[idx] = f"{i}{scale}"
          idx += 1

  data_tf = {category: {index_mapping[i+1]: 0 for i in range(num_scale)} for category in category_instancesonly}
  data_t = {category: {index_mapping[i+1]: 0 for i in range(num_scale)} for category in category_instancesonly}

  # Prompt Tuning tf
  for i in range(len(Prompt_vectors_tf)):
      for k in range(len(Prompt_vectors_tf[0])):
          index_char = index_mapping.get(k + 1, k + 1)  # Gets the mapping value, or returns the original index if it does not exist
          data_tf[category_instancesonly[i]][index_char] = len(Prompt_vectors_tf[i][k])

  # Prompt Tuning t
  for i in range(len(Prompt_vectors_t)):
      for k in range(len(Prompt_vectors_t[0])):
          index_char = index_mapping.get(k + 1, k + 1)  # Gets the mapping value, or returns the original index if it does not exists
          data_t[category_instancesonly[i]][index_char] = len(Prompt_vectors_t[i][k])

  df_tf = pd.DataFrame(data_tf).transpose()
  df_t = pd.DataFrame(data_t).transpose()

  print("Prompt Vectors tf: "+str(total_tf)+" features.")
  print(df_tf.to_string())
  print("\nPrompt Vectors t: "+str(total_t)+" features.")
  print(df_t.to_string())

# Save the query feature center store
def prompt_save(Prompt_vectors,target_device,save_dir):
  # move to device
  for i in range(len(Prompt_vectors)):
    for j in range(len(Prompt_vectors[i])):
        for k in range(len(Prompt_vectors[i][j])):
            Prompt_vectors[i][j][k] = Prompt_vectors[i][j][k].to(target_device)
  ave_query_list_tf = [[[] for _ in range(num_scale)] for _ in range(num_class)]
  total=0
  # compute average
  for i in range(len(Prompt_vectors)):
    for j in range(len(Prompt_vectors[0])):
        if Prompt_vectors[i][j]:
            total=total+len(Prompt_vectors[i][j])
            stacked_tensors = torch.stack(Prompt_vectors[i][j])
            average = torch.mean(stacked_tensors, dim=0)
            ave_query_list_tf[i][j] = average
        else:
            ave_query_list_tf[i][j] = torch.zeros(256).to(target_device)
  saved_dict_tf = {}
  # save .pth
  for i in range(len(ave_query_list_tf)):
      saved_dict_tf[i] = {}
      for j in range(len(ave_query_list_tf[i])):
          saved_dict_tf[i][j] = ave_query_list_tf[i][j]
  # Save Average（important）
  torch.save(saved_dict_tf, save_dir)
  return total

# Get the id of the GPU
def get_gpu_id():
    if dist.is_initialized():
        gpu_id = dist.get_rank()
    else:
        gpu_id = torch.cuda.current_device()
    return gpu_id
  
# The save feature is pkl
def save_file_to_pkl(prompts, filename):
  with open(filename, 'wb') as f:
      pickle.dump(prompts, f)

# read
def load_prompts_from_file(filename):
  with open(filename, 'rb') as f:
      prompts = pickle.load(f)
  return prompts

# Integrate features on all Gpus
def merge_prompts_files(file_list_t):
    Prompt_vectors = [[] for _ in range(num_class)]
    for file_path in file_list_t:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            for i in range(len(Prompt_vectors)):
                Prompt_vectors[i].extend(data[i])
    return Prompt_vectors

# Integrate feature scale statistics on all Gpus
def merge_ar_files(file_list):
    merged_area_ratios = [[] for _ in range(num_class)]
    for file in file_list:
        area_ratios= load_prompts_from_file(file)
        for i in range(num_class):
            merged_area_ratios[i].extend(area_ratios[i])
    return merged_area_ratios

# move to cpu
def move_cpu(Prompt_vectors):
  for i in range(num_class):
      for j in range(num_scale):
          for k in range(len(Prompt_vectors[i][j])):
              Prompt_vectors[i][j][k] = Prompt_vectors[i][j][k].cpu()
  return Prompt_vectors

# Remove duplicate content
def delete_repetion(result_pt):
  boxes = result_pt['boxes']
  cxcywh = result_pt['cxcywh']
  labels = result_pt['labels']
  scores = result_pt['scores']
  id_query = result_pt['id_query']

  unique_ids, counts = id_query.unique(return_counts=True)
  repeated_ids = unique_ids[counts > 1]

  keep_indices = []
  for id in repeated_ids:
      indices = (id_query == id).nonzero(as_tuple=True)[0]
      max_score_index = indices[scores[indices].argmax()]
      keep_indices.append(max_score_index.item())

  unique_only_ids = unique_ids[counts == 1]
  for id in unique_only_ids:
      index = (id_query == id).nonzero(as_tuple=True)[0].item()
      keep_indices.append(index)

  keep_indices = sorted(list(set(keep_indices)))

  filtered_boxes = boxes[keep_indices]
  filtered_cxcywh = cxcywh[keep_indices]
  filtered_labels = labels[keep_indices]
  filtered_scores = scores[keep_indices]
  filtered_id_query = id_query[keep_indices]

  result_pt['boxes'] = filtered_boxes
  result_pt['cxcywh'] = filtered_cxcywh
  result_pt['labels'] = filtered_labels
  result_pt['scores'] = filtered_scores
  result_pt['id_query'] = filtered_id_query
  return result_pt

# convert cxcywh to xyxy
def convert_cxcywh_to_xyxy(boxes, image_size):
    """
    Convert normalized cxcywh boxes to xyxy format.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) with normalized cxcywh format.
        image_size (tuple): Tuple of (height, width) of the image.

    Returns:
        xyxy_boxes (torch.Tensor): Tensor of shape (N, 4) with xyxy format.
    """
    h, w = image_size
    cxcywh_boxes = boxes.clone()
    cxcywh_boxes[:, 0] *= w  # cx
    cxcywh_boxes[:, 1] *= h  # cy
    cxcywh_boxes[:, 2] *= w  # width
    cxcywh_boxes[:, 3] *= h  # height

    x1 = cxcywh_boxes[:, 0] - cxcywh_boxes[:, 2] / 2
    y1 = cxcywh_boxes[:, 1] - cxcywh_boxes[:, 3] / 2
    x2 = cxcywh_boxes[:, 0] + cxcywh_boxes[:, 2] / 2
    y2 = cxcywh_boxes[:, 1] + cxcywh_boxes[:, 3] / 2

    xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return xyxy_boxes

# convert xyxy to cxcywh
def convert_xyxy_to_cxcywh(boxes, image_size):
    """
    Convert xyxy boxes to normalized cxcywh format.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) with xyxy format.
        image_size (tuple): Tuple of (height, width) of the image.

    Returns:
        cxcywh_boxes (torch.Tensor): Tensor of shape (N, 4) with normalized cxcywh format.
    """
    h, w = image_size
    xyxy_boxes = boxes.clone()

    cx = (xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2
    cy = (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2
    width = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]
    height = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]

    cxcywh_boxes = torch.stack([cx, cy, width, height], dim=1)

    cxcywh_boxes[:, 0] /= w  # cx
    cxcywh_boxes[:, 1] /= h  # cy
    cxcywh_boxes[:, 2] /= w  # width
    cxcywh_boxes[:, 3] /= h  # height

    return cxcywh_boxes

# OSS
def custom_nms(scores, labels, boxes, image_size, iou_threshold=0.5, device='cpu'):
    """
    Custom NMS that considers class labels and keeps the box with the highest score if its IoU with
    another box of the same class is greater than the threshold. If a box has no overlapping boxes
    or IoU less than the threshold, it is discarded.

    Args:
        scores (torch.Tensor): Tensor of shape (N,) containing the scores of the boxes.
        labels (torch.Tensor): Tensor of shape (N,) containing the class labels of the boxes.
        boxes (torch.Tensor): Tensor of shape (N, 4) containing the coordinates of the boxes in cxcywh format.
        image_size (tuple): Tuple of (height, width) of the image.
        iou_threshold (float): IoU threshold to determine overlap.
        device (str): Device to perform the computation on.

    Returns:
        final_scores (torch.Tensor): Scores of the kept boxes.
        final_labels (torch.Tensor): Labels of the kept boxes.
        final_boxes (torch.Tensor): Coordinates of the kept boxes.
        mask (torch.Tensor): Boolean mask of the kept boxes.
    """
    keep = []

    if len(boxes) == 0:
        return scores, labels, boxes, torch.tensor(keep, dtype=torch.bool)

    # Convert cxcywh to xyxy format
    boxes = convert_cxcywh_to_xyxy(boxes, image_size)

    # Move to device
    scores = scores.to(device)
    labels = labels.to(device)
    boxes = boxes.to(device)

    unique_labels = labels.unique()
    for label in unique_labels:
        # Filter boxes, scores and labels of the current class
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]

        # Compute pairwise IoU
        iou = box_iou(class_boxes, class_boxes)

        # Sort the boxes by scores in descending order
        sorted_indices = torch.argsort(class_scores, descending=True)

        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            sorted_indices = sorted_indices[1:]

            # Find boxes with IoU > iou_threshold
            high_iou_indices = (iou[current, sorted_indices] > iou_threshold).nonzero(as_tuple=True)[0]

            if len(high_iou_indices) > 0:
                # Keep the current box (it has the highest score)
                keep.append(class_indices[current].item())

                # Remove all boxes with high IoU
                sorted_indices = sorted_indices[~(iou[current, sorted_indices] > iou_threshold)]
            else:
                # Remove the current box (no significant overlap)
                pass

    keep = torch.tensor(keep, dtype=torch.long, device=device)
    mask = torch.zeros(len(scores), dtype=torch.bool, device=device)
    mask[keep] = True

    final_scores = scores[keep]
    final_labels = labels[keep]
    final_boxes = boxes[keep]
    final_boxes = convert_xyxy_to_cxcywh(final_boxes, image_size)

    return final_scores, final_labels, final_boxes, mask

# Compute the IoU of two sets of boxes.
def box_iou(boxes1, boxes2):
    """
    Compute the IoU of two sets of boxes.

    Args:
        boxes1 (torch.Tensor): Tensor of shape (N, 4).
        boxes2 (torch.Tensor): Tensor of shape (M, 4).

    Returns:
        iou (torch.Tensor): Tensor of shape (N, M) containing the pairwise IoU values.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou
  
# The few labels with the highest confidence
def learn_tops(result_pt, learn_thre, top_num):
  for bs in range(len(result_pt)):
    if result_pt[bs] is not None:
        scores = result_pt[bs]['scores']
        labels = result_pt[bs]['labels']
        unique_labels = torch.unique(labels)
        
        final_indices = []

        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            label_scores = scores[label_indices]

            high_conf_indices = label_indices[(label_scores > learn_thre).nonzero(as_tuple=True)[0]]

            if len(high_conf_indices) > top_num:
                top_indices = torch.topk(scores[high_conf_indices], top_num).indices
                high_conf_indices = high_conf_indices[top_indices]

            final_indices.append(high_conf_indices)

        final_indices = torch.cat(final_indices)

        result_pt[bs]['boxes'] = result_pt[bs]['boxes'][final_indices]
        result_pt[bs]['cxcywh'] = result_pt[bs]['cxcywh'][final_indices]
        result_pt[bs]['labels'] = result_pt[bs]['labels'][final_indices]
        result_pt[bs]['scores'] = result_pt[bs]['scores'][final_indices]
        result_pt[bs]['id_query'] = result_pt[bs]['id_query'][final_indices]

  return result_pt

# iou
def box_iou_cxcywh(box1, box2):
    def cxcywh_to_xyxy1(box):
        cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
    
    box1 = cxcywh_to_xyxy1(box1)
    box2 = cxcywh_to_xyxy1(box2)
    
    inter_xmin = torch.max(box1[:, 0], box2[:, 0])
    inter_ymin = torch.max(box1[:, 1], box2[:, 1])
    inter_xmax = torch.min(box1[:, 2], box2[:, 2])
    inter_ymax = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = (inter_xmax - inter_xmin).clamp(0) * (inter_ymax - inter_ymin).clamp(0)
    
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

# calculate area ratios
def calculate_area_ratios(learn,epoch,is_t):
    if is_t:
      for bs in range(len(learn)):
        if learn[bs]!=None:
          boxes = learn[bs]['boxes']
          labels = learn[bs]['labels']
          areas = boxes[:, 2] * boxes[:, 3]
          for i in range(len(labels)):
              category = labels[i].item()-1
              area_ratios_t[category].append([areas[i].item(),epoch])
    else:
      for bs in range(len(learn)):
        if learn[bs]!=None:
          boxes = learn[bs]['boxes']
          labels = learn[bs]['labels']
          areas = boxes[:, 2] * boxes[:, 3]
          for i in range(len(labels)):
              category = labels[i].item()-1
              area_ratios_tf[category].append([areas[i].item(),epoch])

# Statistical average segmentation of scales
def interval_seg(area_ratios_t,is_t):
    results = []
    for ratios in area_ratios_t:
        if ratios != []:
            sorted_ratios = sorted(ratios)
            num_bins = num_scale
            quantiles = np.linspace(0, 1, num_bins + 1)
            bins = np.quantile(sorted_ratios, quantiles)
            bins[0],bins[num_scale]=0.0,1.0
            count = np.histogram(sorted_ratios, bins=bins)[0]
        else:
            bins = np.zeros(6)
            count = np.zeros(num_scale)
        results.append({
            'intervals': bins,
            'counts': count
        })
    return results

# Segmentation features according to scale statistics
def interval_all(Prompt_vectors,inter,burn_in,epoch):
  allocated_prompts= [[[] for _ in range(num_scale)] for _ in range(num_class)]
  for i in range(len(Prompt_vectors)):
    if i<len(inter):
      intervals = inter[i]['intervals']
      for j in range(len(Prompt_vectors[i])):
          if burn_in:
            # Get area_ratio and features
            area_ratio = Prompt_vectors[i][j]['area_ratio']
            features = Prompt_vectors[i][j]['features']
            index = np.searchsorted(intervals, area_ratio) - 1
            allocated_prompts[i][int(index)].append(features)
          else:
            if Prompt_vectors[i][j]['epoch'] in {epoch}:
              # Get area_ratio and features
              area_ratio = Prompt_vectors[i][j]['area_ratio']
              features = Prompt_vectors[i][j]['features']
              index = np.searchsorted(intervals, area_ratio) - 1
              allocated_prompts[i][int(index)].append(features)
  return allocated_prompts

# Remove redundant boxes
def process_tf_labels(t_tf_label):
    processed_labels = []

    for batch in t_tf_label:
      if batch['labels'].shape[0]==0:
        processed_labels.append(batch)
        continue
      else:
        boxes=batch['boxes']
        cxcywh = batch['cxcywh']
        labels = batch['labels']
        scores = batch['scores']
        id_query = batch['id_query']

        keep_indices = list(range(len(cxcywh)))  
        
        unique_labels = labels.unique()
        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            
            label_cxcywh = cxcywh[label_indices]
            label_scores = scores[label_indices]

            for i in range(len(label_cxcywh)):
                for j in range(i + 1, len(label_cxcywh)):
                    box1 = label_cxcywh[i]
                    box2 = label_cxcywh[j]

                    x1_1 = box1[0] - box1[2] / 2
                    y1_1 = box1[1] - box1[3] / 2
                    x2_1 = box1[0] + box1[2] / 2
                    y2_1 = box1[1] + box1[3] / 2
                    
                    x1_2 = box2[0] - box2[2] / 2
                    y1_2 = box2[1] - box2[3] / 2
                    x2_2 = box2[0] + box2[2] / 2
                    y2_2 = box2[1] + box2[3] / 2
                    
                    intersection_x1 = max(x1_1, x1_2)
                    intersection_y1 = max(y1_1, y1_2)
                    intersection_x2 = min(x2_1, x2_2)
                    intersection_y2 = min(y2_1, y2_2)

                    intersection_width = max(0, intersection_x2 - intersection_x1)
                    intersection_height = max(0, intersection_y2 - intersection_y1)
                    intersection_area = intersection_width * intersection_height

                    area1 = box1[2] * box1[3]
                    area2 = box2[2] * box2[3]
                    
                    if area1 > 0 and area2 > 0:
                        overlap_ratio = max(intersection_area / area1,intersection_area / area2)

                        if overlap_ratio > 0.9:
                            if label_scores[i] >= label_scores[j]:
                              if label_indices[j].item() in keep_indices:
                                keep_indices.remove(label_indices[j].item())
                            else:
                              if label_indices[i].item() in keep_indices:
                                keep_indices.remove(label_indices[i].item())

        processed_batch = {
            'boxes': boxes[keep_indices],
            'cxcywh': cxcywh[keep_indices],
            'labels': labels[keep_indices],
            'scores': scores[keep_indices],
            'id_query': id_query[keep_indices],
            'size': batch['size'],
            'orig_size': batch['orig_size']
        }
        
        processed_labels.append(processed_batch)

    return processed_labels
