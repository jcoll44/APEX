import torch
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

def average_segcover(segA, segB, scale=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    
    #segA = torch.Tensor(np.argmax(segA, axis=-1)).view(1,1,128,128).long()
    #segB = torch.Tensor(np.argmax(segB, axis=-1)).view(1,1,128,128).long()
    assert segA.shape == segB.shape
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    N = torch.tensor(bsz*[0])
    scores = torch.tensor(bsz*[0.0])
    # Loop over segA
    for i in range(segA.max().item() + 1):
        binaryA = segA == i
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in range(segB.max().item() + 1):
            binaryB = segB == j
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Scale
        if scale:
            scores += binaryA.sum((1, 2, 3)).float() * max_iou
        else:
            scores += max_iou
    if scale:
        nonignore = segA >= 0
        coverage = scores / nonignore.sum((1, 2, 3)).float()
    else:
        coverage = scores / N.float()
    # Sanity check
    assert (0.0 <= coverage).all() and (coverage <= 1.0).all()
    # Take average over batch dimension
    coverage = coverage.mean(0).item()
    return coverage

def ari_score2(masks_true, masks_pred):
    clustering_masks_true = np.argmax(masks_true, axis=-1).flatten()
    clustering_masks = np.argmax(masks_pred, axis=-1).flatten()
    return adjusted_rand_score(clustering_masks_true, clustering_masks)

def get_gt_masks(gt_dir,trj_idx,bs):
    masks_gt_all = []
    masks_gt_bg_all = []
    masks_gt_fg_all = []
    for b in range(bs):
        which_trj = trj_idx[b]
        masks_list = np.load(gt_dir+'{}.npy'.format(int(which_trj)),allow_pickle=True)
        masks_gt_i = []
        masks_gt_bg_i = []
        masks_gt_fg_i = []
        for i in range(20):
            mask_gt=np.stack(masks_list[i]['masks'],axis=2)#128,128,m
            masks_gt_i.append(mask_gt)
            mask_gt_bg=mask_gt[:,:,0]
            masks_gt_bg_i.append(mask_gt_bg)
            mask_gt_fg=mask_gt[:,:,1:]
            masks_gt_fg_i.append(mask_gt_fg)
        masks_gt_all.append(masks_gt_i)
        masks_gt_bg_all.append(masks_gt_bg_i)
        masks_gt_fg_all.append(masks_gt_fg_i)
    return masks_gt_all,masks_gt_fg_all,masks_gt_bg_all

def binarize_masks(masks):
    ''' Binarize soft masks.
    Args:
        masks: torch.Tensor(CxHxW)
    '''
    n = masks.size(0)
    idc = torch.argmax(masks, axis=0)
    binarized_masks = torch.zeros_like(masks)
    for i in range(n):
        binarized_masks[i] = (idc == i).int()
    return binarized_masks


def calculate_iou(mask1, mask2):
    ''' Calculate IoU of two segmentation masks.
    Args:
        mask1: HxW
        mask2: HxW
    '''
    eps = np.finfo(float).eps
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)
    union = ((np.sum(mask1) + np.sum(mask2) - np.sum(mask1*mask2)))
    iou = np.sum(mask1*mask2) / (union + eps)
    iou = 1. if union == 0. else iou
    return iou 


def compute_mot_metrics(acc, summary):
    ''' Args:
            acc: motmetric accumulator
            summary: pandas dataframe with mometrics summary
    '''
    df = acc.mot_events
    df = df[(df.Type != 'RAW')
            & (df.Type != 'MIGRATE')
            & (df.Type != 'TRANSFER')
            & (df.Type != 'ASCEND')]
    obj_freq = df.OId.value_counts()
    n_objs = len(obj_freq)
    tracked = df[df.Type == 'MATCH']['OId'].value_counts()
    detected = df[df.Type != 'MISS']['OId'].value_counts()

    track_ratios = tracked.div(obj_freq).fillna(0.)
    detect_ratios = detected.div(obj_freq).fillna(0.)

    summary['mostly_tracked'] = track_ratios[track_ratios >= 0.8].count() / n_objs * 100
    summary['mostly_detected'] = detect_ratios[detect_ratios >= 0.8].count() / n_objs * 100

    n = summary['num_objects'][0]
    summary['num_matches']  = (summary['num_matches'][0] / n * 100)
    summary['num_false_positives'] = (summary['num_false_positives'][0] / n * 100)
    summary['num_switches'] = (summary['num_switches'][0] / n * 100)
    summary['num_misses']  = (summary['num_misses'][0] / n * 100)
    
    summary['mota']  = (summary['mota'][0] * 100)
    summary['motp']  = ((1. - summary['motp'][0]) * 100)

    return summary


def rle_encode(img):
    '''
    from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

 
def decode_rle(mask_rle, shape):
    '''
    from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
