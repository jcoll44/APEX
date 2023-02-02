import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import motmetrics as mm

from eval_utils import calculate_iou, decode_rle, compute_mot_metrics
parser = argparse.ArgumentParser(description='APEX')
parser.add_argument('--ex_id', default=0,
                    help='which experiment to eval.')
args = parser.parse_args()
gt_dict = np.load('./data/P2D/annotation/tracking/gt_test.npy',allow_pickle=True)
pred_dict = np.load(f'./tracking_results/P2D/{args.ex_id}/pred.npy',allow_pickle=True)

def exclude_bg(dists, gt_ids, pred_ids, n_gt_bg=1):
    # remove background slots
    gt_idx = -1
    for k in range(n_gt_bg):
        if dists.shape[1] > 0:
            pred_bg_id = np.where(dists[gt_idx] > 0.2)[0]
            dists = np.delete(dists, pred_bg_id, 1)
            pred_ids = [pi for l, pi in enumerate(pred_ids) if not l in pred_bg_id]
        dists = np.delete(dists, gt_idx, 0)  
        del gt_ids[gt_idx]   
    return dists, gt_ids, pred_ids

def accumulate_events(gt_dict, pred_dict):
    acc = mm.MOTAccumulator()
    count = 0
    for i in tqdm(range(len(gt_dict))):
        for t in range(20):
            gt_dict_frame = gt_dict[i][t]
            pred_dict_frame = pred_dict[i][t]
            dist, gt_ids, pred_ids = compute_dists_per_frame(gt_dict_frame, pred_dict_frame)
            acc.update(gt_ids, pred_ids, dist, frameid=count)
            count += 1
    return acc

def compute_dists_per_frame(gt_frame, pred_frame):
    # Compute pairwise distances between gt objects and predictions per frame. 
    s = 128
    n_pred = len(pred_frame['ids'])
    n_gt = len(gt_frame['ids'])

    # accumulate pred masks for frame
    preds = []
    pred_ids = []
    for j in range(n_pred):
        mask = decode_rle(pred_frame['masks'][j], (s, s))
        if mask.sum() > 5:
            preds.append(mask)
            pred_ids.append(pred_frame['ids'][j])
    preds = np.array(preds)

    # accumulate gt masks for frame
    gts = []
    gt_ids = []
    for h in range(n_gt):
        mask = decode_rle(gt_frame['masks'][h], (s, s))
        if mask.sum() > 5:
            gts.append(mask)
            gt_ids.append(gt_frame['ids'][h])
    gts = np.array(gts)

    # compute pairwise distances
    dists = np.ones((len(gts), len(preds)))
    for h in range(len(gts)):
        for j in range(len(preds)): 
            dists[h, j] = calculate_iou(gts[h], preds[j])
    
    dists, gt_ids, pred_ids = exclude_bg(dists, gt_ids, pred_ids, 1)
    dists = 1. - dists
    dists[dists > 0.5] = np.nan
        
    return dists, gt_ids, pred_ids

# start eval.
count_id=0
for i in range(200):
    video=pred_dict[i]
    count_id=320*i#make id in each video unique, maximum possible num of objs is 20*16(cell num.)
    for t in range(20):
        frame=video[t]
        obj_num=len(frame['ids'])
        pred_dict[i][t]['ids']=[x+count_id for x in frame['ids']]
assert len(pred_dict) == len(gt_dict)

print('Acuumulate events.')
acc = accumulate_events(gt_dict, pred_dict)

print('Accumulate metrics.')
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses', 'num_objects'], name='acc')

# compute tracking metrics
metrics = compute_mot_metrics(acc, summary)

print(metrics)
metrics = {key: value['acc'] for key, value in metrics.items()}
metrics = pd.Series(metrics).to_json()
#_scalor
print('Saving results to {}'.format(f'./results{args.ex_id}_APEX.json'))
with open(f'./tracking_results/P2D/{args.ex_id}/results{args.ex_id}_APEX.json', 'w') as f:
    json.dump(metrics, f)

print('Done.')


