import urllib.request
import os


# default_data_dir = <project_root>/data
# resnet models: = <project_root>/data/models/resnet/<weights_file>.h5
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data"))


def download_ResNet152_weights_tf(data_dir=DEFAULT_DATA_DIR):
    tf_url = "https://doc-08-8g-docs.googleusercontent.com/docs/securesc/ctnhkpgoq04mmd8j7e1mca93ov7uclmi/" \
            "vujvdr77b4drs5jutkj3ghbtl9cgb5nd/1530172800000/09552578107191760921/" \
            "02683311644763403734/0Byy2AcGyEVxfeXExMzNNOHpEODg?e=download&nonce=34km3qqnfadgi&" \
            "user=02683311644763403734&hash=5gjtkf7q0e75ace8t5f36g9vgm11nutb"
    filename = "resnet152_weights_tf.h5"
    file_path = os.path.join(data_dir, "models", "resnet", filename)

    _download_if_not_exist(tf_url, file_path)


def download_ResNet152_weights_th(data_dir=DEFAULT_DATA_DIR):
    th_url = "https://doc-14-8g-docs.googleusercontent.com/docs/securesc/ctnhkpgoq04mmd8j7e1mca93ov7uclmi/" \
            "j6srcrlctejmp5fhrlbmefrhsj5ldbbc/1530172800000/09552578107191760921/" \
            "02683311644763403734/0Byy2AcGyEVxfZHhUT3lWVWxRN28?e=download"
    filename = "resnet152_weights_th.h5"
    file_path = os.path.join(data_dir, "models", "resnet", filename)

    _download_if_not_exist(th_url, file_path)


def _download_if_not_exist(url, file_path):
    if os.path.isfile(file_path):
        print("{} does already exist, skipping download".format(file_path))
        return

    else:
        print("downloading resnet-152 model weights to {}".format(file_path))
        urllib.request.urlretrieve(url, file_path)
