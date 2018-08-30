DL-Image-Captioning
===================
Project for "Competitive Problem Solving with Deep Learning" at the Hasso-Plattner Institute.

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

### Use with Docker

**tbd**

### Use locally

#### Training

#### Prediction

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


## Known Problems

- If you want to use METEOR scores, the whole repo must be located in a path without spaces.
  The METEOR scorer is written in Java and creates a file path URL relative to its location pointing to a data file.
  The file path is urlencoded twice in the Java code and therefore leads to an exception because it can not find the data file.
  This can only be avoided by not using any space or other special characters in the file path to the JAR file of the scorer.

- METEOR scoring currently does not work, because of subprocess communication issues with the Java code.
