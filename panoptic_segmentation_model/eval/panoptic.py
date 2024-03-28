import functools
import traceback
from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor


class PanopticEvaluator:
    """
    Evaluate panoptic segmentation
    """

    def __init__(self, stuff_list: List[int], thing_list: List[int], label_divisor: int = 1000,
                 void_label: int = -1):
        self.stuff_list = stuff_list
        self.thing_list = thing_list
        self.label_divisor = label_divisor
        self.void_label = void_label
        self.pq_stats = PQStat()

    def _get_meta_from_panoptic(self, pan_img: Tensor) -> List:
        segments_info = []
        batch_size = pan_img.shape[0]
        for b in range(batch_size):
            pan_img_b = pan_img[b]
            segments_info_b = []
            pan_label, pan_area = torch.unique(pan_img_b, return_counts=True)
            for label, area in zip(pan_label, pan_area):
                if label == self.void_label:
                    continue
                pred_class = int(torch.div(label, self.label_divisor, rounding_mode="floor"))
                segments_info_b.append(
                    {"id": int(label), "category_id": pred_class, "area": int(area)})
            segments_info.append(segments_info_b)
        return segments_info

    def update(self, pan_img_gt: Tensor, pan_img_pred: Tensor):
        meta_pred = self._get_meta_from_panoptic(pan_img_pred)
        meta_gt = self._get_meta_from_panoptic(pan_img_gt)
        pq_i = compute_panoptic_stats(meta_gt, meta_pred, pan_img_gt, pan_img_pred, self.void_label)

        self.pq_stats += pq_i

    def evaluate(self) -> Dict[str, dict]:
        results = {
            "Things": self.pq_stats.pq_average(self.thing_list)[0],
            "Stuff": self.pq_stats.pq_average(self.stuff_list)[0]
        }
        all_list = self.stuff_list + self.thing_list
        results["All"], results["per_class"] = self.pq_stats.pq_average(all_list)
        return results

    def reset(self):
        self.pq_stats = PQStat()


# -------------------------------------------------------- #

class PQStatCat:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat:
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, index: int):
        return self.pq_per_cat[index]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories: List[int]):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}

        for label in categories:
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1

            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        if n == 0:
            return {"pq": 0, "sq": 0, "rq": 0, "n": n}, per_class_results
        return {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}, per_class_results


# -------------------------------------------------------- #

# The decorator is used to print an error thrown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("Caught exception in worker thread:")
            traceback.print_exc()
            raise e

    return wrapper


@get_traceback
def compute_panoptic_stats(meta_gt, meta_pred, pan_img_gt, pan_img_pred, void_label: int = -1):
    OFFSET = 256 * 256 * 256
    pq_stat = PQStat()

    # Iterate through the whole batch
    batch_size = pan_img_gt.shape[0]
    for b in range(batch_size):
        pan_img_gt_b = pan_img_gt[b]
        pan_img_pred_b = pan_img_pred[b]
        meta_gt_b = meta_gt[b]
        meta_pred_b = meta_pred[b]

        gt_segms = {el["id"]: el for el in meta_gt_b}
        pred_segms = {el["id"]: el for el in meta_pred_b}

        # Find the matching between the GT instance IDs and the predicted IDs
        # We need a positive void label for the following trick to work.
        tmp_void_label = 19 * 1000  # This is greater than the largest expected ID 180xx.
        pan_img_gt_b[pan_img_gt_b == void_label] = tmp_void_label
        pan_img_pred_b[pan_img_pred_b == void_label] = tmp_void_label
        pan_gt_pred = pan_img_gt_b.to(torch.int64) * OFFSET + pan_img_pred_b.to(torch.int64)
        pan_img_gt_b[pan_img_gt_b == tmp_void_label] = void_label
        pan_img_pred_b[pan_img_pred_b == tmp_void_label] = void_label
        gt_pred_map = {}
        labels, labels_cnt = torch.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = int(label) // OFFSET
            pred_id = int(label % OFFSET)
            gt_id = void_label if gt_id == tmp_void_label else gt_id
            pred_id = void_label if pred_id == tmp_void_label else pred_id
            gt_pred_map[(gt_id, pred_id)] = int(intersection)

        # Count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if void_label in label_tuple:
                continue
            if gt_label not in gt_segms:
                assert False, gt_label
            if pred_label not in pred_segms:
                assert False, pred_label
            if gt_segms[gt_label]["category_id"] != pred_segms[pred_label]["category_id"]:
                continue

            union = pred_segms[pred_label]["area"] \
                    + gt_segms[gt_label]["area"] \
                    - intersection \
                    - gt_pred_map.get((void_label, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]["category_id"]].tp += 1
                pq_stat[gt_segms[gt_label]["category_id"]].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # Count false negatives
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            pq_stat[gt_info["category_id"]].fn += 1

        # Count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((void_label, pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to
            # VOID and CROWD regions
            if intersection / pred_info["area"] > 0.5:
                continue
            pq_stat[pred_info["category_id"]].fp += 1
    return pq_stat
