import os; import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.common.evaluation.pycocoevalcap.eval import COCOEvalCap
from src.common.evaluation.pycocotools.coco import COCO

import subprocess
import argparse


def evaluate(gt_path, prediction_path):
    # get stanford nltk data
    subprocess.call(['./get_stanford_models.sh'])

    # create ground truth coco object and load results
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(prediction_path)

    # consider only n images
    n = 10
    img_ids = sorted(coco_gt.getImgIds())
    img_ids = img_ids[0:n]

    # evaluate results
    coco_eval = COCOEvalCap(coco_gt, coco_dt)
    coco_eval.params['img_id'] = img_ids
    coco_eval.evaluate()

    # for now print results
    for metric, score in coco_eval.eval.items():
        print('%s: %.3f' % (metric, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses the coco tools to calculate MS COCO scores for ' +
                                                 'ground truth data and predictions.')

    parser.add_argument('ground_truth',
                        type=str,
                        help="filepath to the ground truth data in JSON format")
    parser.add_argument('predictions',
                        type=str,
                        help="filepath to the predicted captions in JSON format")

    arguments = parser.parse_args()
    evaluate(arguments.ground_truth, arguments.predictions)
