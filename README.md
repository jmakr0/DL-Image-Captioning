DL-Image-Captioning
===================
Project for "Competitive Problem Solving with Deep Learning" at the Hasso-Plattner Institute.
This repository contains the Keras source code for our four different image captioning models:

| Model | Keras Model Definition | Model Plot |
| :---  | :--------------------- | :--------- |
| Model 1 (base model) | [src/cacao/model_raw.py](src/cacao/model_raw.py) | [Open Plot](doc/model_plots/model_raw.pdf) |
| Model 2 (image loop) | [src/cacao/model_image_loop.py](src/cacao/model_image_loop.py) | [Open Plot](doc/model_plots/model_image_loop.pdf) |
| Model 3 (attention) | [src/cacao/model.py](src/cacao/model.py) | [Open Plot](doc/model_plots/model_full.pdf) |
| Model 4 (softmax classification) | [src/cacao/model_softmax.py](src/cacao/model_softmax.py) | [Open Plot](doc/model_plots/model_softmax.pdf) |

We developed these model with Python 3.6.3, Keras 2.2.0 and the TensorFlow 1.8.0 backend.

## Requirements

These are the requirements for local usage and development.
You can also use a Docker image to make predictions.
See section [Use with Docker](#use-with-docker) below.

**OS-level requirements**

- java 1.8.0
- python 3

**Python packages:**

See file `requirements.txt` for all python packages with their needed version.
Install all the requirements with:

```bash
pip install -r requirements.txt
```

## Usage

You can use this project in two different ways.
First of all there is a Docker-Image available, which can be used to make predictions.
See the next section for details of that.
Second, you can run the code locally and train your own model.
This is described in the section [Use locally](#use-locally).

### Use with Docker

**tbd**

### Use locally

Please make sure, that you have installed all requirements on your machine.
You can always use [virtualenv](https://virtualenv.pypa.io/en/stable/) or an virtualenv-wrapper of your choice to install all python-dependencies.

If you have a GPU at hand and want to use it, please follow the [Keras installation instruction for the use with TensorFlow](https://keras.io/#installation) to setup your environment.
Don't forget to install cuDNN as well.

For the next sections we assume that all commands are run from the project's root directory.
Every path that can be specified in script arguments must exist before running the script.

#### Training

If you want to train the models with your own data, you can find the `train.py` script in the `src`-folder: [train.py](src/train.py).
It provides a help-page that can be displayed via:

```
$> python src/train.py --help
usage: train.py [-h] [--exp_id EXP_ID] [--devices [DEVICES [DEVICES ...]]]
                [--cnn {resnet50,resnet152}] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--lr LR] [--workers WORKERS]
                [--model_type {full,image_loop,raw,softmax}]
                settings

Starts the training of the `cacao` model

positional arguments:
  settings              filepath to the configuration file

optional arguments:
  -h, --help            show this help message and exit
  --exp_id EXP_ID       experiment ID for saving logs (when empty: timestamp)
  --devices [DEVICES [DEVICES ...]]
                        IDs of the GPUs to use, starting from 0
  --cnn {resnet50,resnet152}
                        type of CNN to use as image feature extractor
  --epochs EPOCHS       trainings stops after this number of epochs
  --batch_size BATCH_SIZE
                        batch size as power of 2; if multiple GPUs are used
                        the batch is divided between them
  --lr LR               learning rate for the optimization algorithm
  --workers WORKERS     number of worker-threads to use for data preprocessing
                        and loading
  --model_type {full,image_loop,raw,softmax}
                        selects model to train with growing capabilities
```

As you can see, you are required to pass a path to a settings file.
This file is specific to the system the code is running on and can be used to specify to path to various files and folders, including the training and validation data.
In addition it contains information about the images and the word embedding.
See [settings.yml](src/settings/settings.yml) for an settings file example.

> Attention!
>
> Currently our models are not compatible with keras' multi-GPU (`multi_gpu_model`) support.
> This means, you can use the `--devices`-switch only with one GPU ID, subsequent IDs will be trimmed.

#### Prediction

Prediction is implemented in the [`predict.py`-script](src/predict.py).
It also supports the `--help`-switch and requires you to pass three paths: to the model-file, to the output-file and to the settings-file.
Please make sure, you specify the correct model type with `--model_type {full,image_loop,raw,softmax}`, so the pre- and postprocessing is correct for the used model.
The prediction results are written into the specified output-file in Json format (`[{"image_id": 0, "caption": "text"}, ...]`).

You can use `declare +gx CUDA_VISIBLE_DEVICES=0; python src/predict.py <params>` to specify the GPU that should be used for prediction.
If `CUDA_VISIBLE_DEVICES` is not specified, tensorflow will allocate all available memory across all installed GPUs.

#### Evaluation

After generating the prediction as a JSON file, you can evaluate your results with the `eval.py` script.
It uses the MSCOCO evaluation codes:
- Bleu
- Meteor (**currently broken**)
- Rouge-L
- CIDEr
- SPICE

You can find more information about the usage of the script by running
```bash
python src/eval.py --help
```
and about the implemented scores in the [README file](./src/common/evaluation/README.md) of the evaluation module.

#### Helper scripts

This project comes with some scripts that help dealing with the data used for prediction and training of the machine learning models.
You can find those scripts in the folder `src/scripts/`.
All scripts support the switch `--help`, which can be used to find out more about the usage of it.

##### Glove Embedding

This project uses the Glove embedding vectors to convert words to vectors for training and prediction.
The class `Glove` builds a wrapper around the raw vector embeddings and provides lookup methods for accessing vectors and words.
It can be found in the file `src/common/dataloader/glove.py`.
The raw embedding file is not normalized, therefore, we provide a helper script to use different normalization methods:
You can use it like that:

```bash
python src/scripts/glove_normalization.py \
  --type '<studentz|minmax>' \
  orig_glove.txt normalized_glove.txt
```

`studentz` standardizes the vectors to a standard deviation of `1`.
For more information take a look at [Wikipedia: Standard score](https://en.wikipedia.org/wiki/Standard_score).
`minmax` in its default configuration scales all values to the interval `[-1, 1]`

##### Filter Metadata

Some datasets used during this project contained metadata, whose filenames to the pictures were erroneous.
The script `filter_metadata.py` can be used to remove unwanted pictures and annotations from the metadata,
so that no `FileNotFoundException`s are thrown during training.
Unfortunately, you have to find out incorrect image IDs by yourself at the moment.
Usage:

```bash
python src/scripts/filter_metadata.py \
  --negative_image_ids 1234 5678 \
  orig_metadata.json cleaned_metadata.json

```

##### Convert Saved Weights to a Model

During training, you have to options to save your current progress:
Saving the pickled model or just saving the weights.
If you have chosen to just save the weights of a trained model,
you can use the script `weights2model.py` to convert the weights to a pickled model file.
This script can only be used for `cacao` weights that were created with the same model version than the source code.
Use it like that:

```bash
python src/script/weights2model.py weights.h5 model.pkl
```

You can use the flag `-p` to plot the model as a `.png`-picture under the name `cacao_model.png`.
For this to work, you must have installed `pydot` and [Graphviz](https://www.graphviz.org/).

## Development

### Docker

Build the image locally:

``docker build -f docker/Dockerfile . --tag image-captioning``

Run the container:

``docker run --rm -v "Your_INPUT_Directory_Path:/usr/src/app/data/input" -v "Your_OUTPUT_Directory_Path:/usr/src/app/data/output" image-captioning``

Upload the container to dockerhub:

``bash docker/push_to_dockerhub.sh``

### Contribution

You are more than welcome to file issues or submit pull requrests to this repository.

**Developers:**

Please contact Axel Stebner (`axel.stebner(at)student.hpi.uni-potsdam.de`) or Sebastian Schmidl (`sebastian.schmidl(at)student.hpi.uni-potsdam.de`) for any questions.

- [jmakr0](https://github.com/jmakr0)
- Alexander Preuß [alpreu](https://github.com/alpreu)
- Sebastian Schmidl [CodeLionX](https://github.com/CodeLionX)
- Friedrich Schöne [friedrichschoene](https://github.com/friedrichschoene)
- Axel Stebner [xasetl](https://github.com/xasetl)
- [slin96](https://github.com/slin96)


## Known Problems

- If you want to use METEOR scores, the whole repo must be located in a path without spaces.
  The METEOR scorer is written in Java and creates a file path URL relative to its location pointing to a data file.
  The file path is urlencoded twice in the Java code and therefore leads to an exception because it can not find the data file.
  This can only be avoided by not using any space or other special characters in the file path to the JAR file of the scorer.

- METEOR scoring currently does not work, because of subprocess communication issues with the Java code.

- Training and Prediction with multiple GPUs does not work, because of issues with Keras' `multi_gpu_model` and our usage of `Reshape`-Layers.
  See [#42](https://github.com/jmakr0/DL-Image-Captioning/issues/42) for more details.
