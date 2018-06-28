# DL-Image-Captioning
Project for "Competitive Problem Solving with Deep Learning" at the Hasso-Plattner Institute

## Setup
For the current setup please refer to the imgcap-README


## Install Requirements and Usage

```bash
sudo apt install python-dev python-tk
virtualenv new -p python2.7 <venv>
pip install -r requirements.txt
pip install tensorflow-gpu
mkdir -p logs
touch logs/model_train.log
cd imcap
python model.py
```