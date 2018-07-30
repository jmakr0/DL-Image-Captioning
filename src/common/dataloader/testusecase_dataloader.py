from src.common.dataloader.dataloader import TrainSequence

# fix settings file
from src.settings.settings import Settings

WORKER = 3
N = 1000

Settings.FILE = "../../settings/settings-sebastian.yml"

if __name__ == "__main__":
    sequence = TrainSequence(32, input_caption=False)
    iterations = N #len(sequence)


    print("using keras utils enqueuer")
    from keras.utils import OrderedEnqueuer
    enq = OrderedEnqueuer(sequence, use_multiprocessing=True, shuffle=False)
    print("starting enqueuer")
    enq.start(WORKER, 20)

    batches = 0
    while batches < iterations:
        batch = enq.get()
        next(batch)
        batches += 1

        if batches % 100 == 0:
            print("Processed {} batches".format(batches))

    print("stopping enqueuer")
    enq.stop(1.5)
