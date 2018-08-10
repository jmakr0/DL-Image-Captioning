import os; import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.common.evaluation.pycocoevalcap.eval import COCOEvalCap
from src.common.evaluation.pycocotools.coco import COCO
from src.common.evaluation.get_stanford_models import get_stanford_models

import argparse
import json


def evaluate(gt_path, prediction_path, output_path, n=None):
    # get stanford nltk data
    exit_code = get_stanford_models()
    if exit_code != 0:
        raise RuntimeError("Stanford Model Download was not successful. Please check log output.")

    # create ground truth coco object and load results
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.load_res(prediction_path)
    coco_eval = COCOEvalCap(coco_gt, coco_dt)

    # consider only n images
    if n is not None:
        print("restricting number of images to first {} smallest image ids".format(n))
        img_ids = sorted(coco_gt.get_img_ids())
        img_ids = img_ids[0:n]
        coco_eval.params['img_ids'] = img_ids

    # evaluate results
    coco_eval.evaluate()

    # print results
    print("")
    print("Results")
    for metric, score in coco_eval.eval.items():
        print('%s: %.3f' % (metric, score))

    # check output path and write results
    if output_path:
        print("Writing results to file")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.isdir(output_path):
            raise ValueError("Output path is not a directory!")

        result_file = os.path.splitext(os.path.basename(prediction_path))[0]
        eval_imgs_filename = os.path.join(output_path, '{}_eval_detailed.json'.format(result_file))
        eval_filename = os.path.join(output_path, '{}_eval_summary.json'.format(result_file))
        with open(eval_imgs_filename, 'w') as fh:
            json.dump(coco_eval.eval_imgs, fh)
        with open(eval_filename, 'w') as fh:
            json.dump(coco_eval.eval, fh)
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses the coco tools to calculate MS COCO scores for ' +
                                                 'ground truth data and predictions.')

    parser.add_argument('ground_truth',
                        type=str,
                        help="filepath to the ground truth data in JSON format")
    parser.add_argument('predictions',
                        type=str,
                        help="filepath to the predicted captions in JSON format")
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help="If folder is specified, saves the results as json in the files "
                             "$predictions_eval_detailed.json and $predictions_eval_summary.json. "
                             "The folder gets created automatically.")
    parser.add_argument('-n',
                        type=int,
                        default=None,
                        help="number of images to include in the evaluation, starting from beginning")

    arguments = parser.parse_args()
    evaluate(arguments.ground_truth, arguments.predictions, arguments.output, arguments.n)
