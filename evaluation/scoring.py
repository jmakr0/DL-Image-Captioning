from coco.cocoapi.PythonAPI.pycocotools.coco import COCO
from coco.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval

from argparse import ArgumentParser


def score():
    cocoGt = COCO(args.gt_json_path)
    cocoDt = cocoGt.loadRes(args.output_json_path)

    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:10]

    cocoEval = COCOeval(cocoGt, cocoDt)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # https://github.com/cocodataset/cocoapi
    # https://github.com/tylin/coco-caption/


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--output_json_path', default='C:/repo/DL-Image-Captioning/output.json', type=str)
    arg_parse.add_argument('--gt_json_path', default='C:/data/CDL_2018/Evaluation-I/Evaluation-I/input.json',
                           type=str)

    args = arg_parse.parse_args()
    score()
