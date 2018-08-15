__author__ = 'tylin'

from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice


class COCOEvalCap:
    def __init__(self, coco, coco_res):
        self.eval_imgs = []
        self.eval = {}
        self.img2eval = {}
        self.coco = coco
        self.coco_res = coco_res
        self.params = {'img_ids': coco_res.get_img_ids()}

    def evaluate(self):
        img_ids = self.params['img_ids']
        # img_ids = self.coco.get_img_ids()
        gts = {}
        res = {}
        for img_id in img_ids:
            gts[img_id] = self.coco.imgToAnns[img_id]
            res[img_id] = self.coco_res.imgToAnns[img_id]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),  # does currently not work (some subprocess communication issues)
            (Rouge(),  "ROUGE_L"),
            (Cider(),  "CIDEr"),
            (Spice(),  "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % method)
            score, scores = scorer(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.img2eval:
                self.img2eval[imgId] = {}
                self.img2eval[imgId]["image_id"] = imgId
            self.img2eval[imgId][method] = score

    def set_eval_imgs(self):
        self.eval_imgs = [e for i, e in self.img2eval.items()]
