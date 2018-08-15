from __future__ import division

import json
import os
import subprocess
import tempfile

import numpy as np

from .....settings.settings import Settings
from ..scorer import Scorer

# spice.jar is in the same directory as spice.py - provided with source code
SPICE_JAR = 'spice-1.0.jar'


class Spice(Scorer):
    """
    Main Class to compute the SPICE metric 
    """

    @staticmethod
    def float_convert(obj):
        try:
            return float(obj)
        except (ValueError, OverflowError, TypeError):
            return np.nan

    def __init__(self):
        # load settings
        dirs = Settings().get_spice_dirs()

        # set cwd
        self.cwd = os.path.dirname(os.path.abspath(__file__))

        # set temp directory
        self.temp_dir = os.path.join(self.cwd, dirs['tmp_dir'])
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # set cache directory
        self.cache_dir = os.path.join(self.cwd, dirs['cache_dir'])
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __call__(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        img_ids = sorted(gts.keys())
        
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for i_id in img_ids:
            hypo = res[i_id]
            ref = gts[i_id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
                "image_id": i_id,
                "test": hypo[0],
                "refs": ref
            })

        in_file = tempfile.NamedTemporaryFile('w+t', delete=False, dir=self.temp_dir)
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir)
        out_file.close()
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
                     '-cache', self.cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent']
        subprocess.check_call(spice_cmd, cwd=self.cwd)

        # Read and process results
        with open(out_file.name) as data_file:    
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        img_id_to_scores = {}
        spice_scores = []
        for item in results:
            img_id_to_scores[item['image_id']] = item['scores']
            spice_scores.append(Spice.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in img_ids:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in img_id_to_scores[image_id].items():
                score_set[category] = {k: Spice.float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)
        return average_score, scores
