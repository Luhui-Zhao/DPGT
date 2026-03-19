# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder1, ann_file1, ann_file2,transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder1, ann_file1,ann_file2)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            image1, target1,image2, target2 = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            image1, target1,image2, target2 = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids1[idx]
        target1 = {'image_id': image_id, 'annotations': target1}
        target2 = {'image_id': image_id, 'annotations': target2}
        image1, target1,image2, target2 = self.prepare(image1, target1,image2, target2)
        
        if self._transforms is not None:
            image1, target1,image2, target2 = self._transforms(image1, target1,image2, target2)
            # img, target = self._transforms(image1, target1)

        return image1, target1,image2, target2


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image1, target1,image2, target2):
        w, h = image1.size

        image_id = target1["image_id"]
        image_id = torch.tensor([image_id])

        anno = target1["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image1, target,image2, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    
    scales = [ 800]
    max_size = 2048
    scales2_resize = [800]
    scales2_crop = [384, 600]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ]) # Random horizontal flip + random resizing of image + normalization

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),# Random horizontal flip
                T.RandomSelect( # Choose a transformation at random
                    T.RandomResize(scales, max_size=max_size),# Random resizing
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])# Resize + cut + Resize
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),# Random clipping
                    SLT.LightingNoise(),# Addition of light noise
                    SLT.AdjustBrightness(2), # Adjust brightness
                    SLT.AdjustContrast(2),# Adjust contrast
                ]),              

                normalize,# normalization
            ])

        return T.Compose([
            T.RandomHorizontalFlip(), # Random horizontal flip
            T.RandomResize(scales, max_size=max_size),# Random resizing
            normalize,# normalization
        ])

    if image_set in ['val', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])# Resizing + normalization
        
    raise ValueError(f'unknown {image_set}')


def build(image_set, args,domain1,domain2):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # city2foggy
    if args.da_task=="city2foggy":
      #my change labels
      if domain1=='source_r':
        PATHS1 = {
                "train": (root/"city2foggy" , root / "city2foggy/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain1=='source_f':
        PATHS1 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain1=='target_r':
        PATHS1 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain1=='target_f':
        PATHS1 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/tf" / f'{mode}_val.json'),
                }
      
      if domain2=='source_r':
        PATHS2 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain2=='source_f':
        PATHS2 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain2=='target_r':
        PATHS2 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain2=='target_f':
        PATHS2 = {
                "train": (root/"city2foggy"  , root / "city2foggy/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"city2foggy"  , root / "city2foggy/annotations/tf" / f'{mode}_val.json'),
                }
    
    # sim10k2city
    elif args.da_task=="sim10k2city":
      if domain1=='source_r':
        PATHS1 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city"  , root / "Sim10K2city/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain1=='source_f':
        PATHS1 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain1=='target_r':
        PATHS1 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain1=='target_f':
        PATHS1 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tf" / f'{mode}_val.json'),
                }
      
      if domain2=='source_r':
        PATHS2 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city/sim10k"  , root / "Sim10K2city/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain2=='source_f':
        PATHS2 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain2=='target_r':
        PATHS2 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain2=='target_f':
        PATHS2 = {
                "train": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"Sim10K2city" , root / "Sim10K2city/annotations/tf" / f'{mode}_val.json'),
                }
    # city2bdd100k
    elif args.da_task=="city2bdd100k":
      if domain1=='source_r':
        PATHS1 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k"  , root / "city2bdd100k/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain1=='source_f':
        PATHS1 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain1=='target_r':
        PATHS1 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain1=='target_f':
        PATHS1 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tf" / f'{mode}_val.json'),
                }
      
      if domain2=='source_r':
        PATHS2 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sr" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k/sim10k"  , root / "city2bdd100k/annotations/sr" / f'{mode}_val.json'),
                }
      elif domain2=='source_f':
        PATHS2 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sf" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/sf" / f'{mode}_val.json'),
                }
      elif domain2=='target_r':
        PATHS2 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tr" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tr" / f'{mode}_val.json'),
                }
      elif domain2=='target_f':
        PATHS2 = {
                "train": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tf" / f'{mode}_train.json'),
                "val": (root/"city2bdd100k" , root / "city2bdd100k/annotations/tf" / f'{mode}_val.json'),
                }

    # add some hooks to datasets
    aux_target_hacks_list = None
    img_folder1, ann_file1 = PATHS1[image_set]
    img_folder2, ann_file2 = PATHS2[image_set]

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    try:
        fix_size = args.fix_size
    except:
        fix_size = False
    dataset = CocoDetection(img_folder1, ann_file1, ann_file2, 
            transforms=make_coco_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug, args=args), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )

    return dataset

