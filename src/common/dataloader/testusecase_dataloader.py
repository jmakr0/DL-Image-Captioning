# =================================
# This is an example of how to use the class dataloader
# on the server
# =================================
from src.settings.settings import Settings
from src.common.dataloader.dataloader import TrainSequence

if __name__ == "__main__":
    Settings.FILE = '../../settings/settings-sebastian.yml'

    gen = TrainSequence(32)
    print(gen[0][0].shape)
    print(gen[0][1].shape)
