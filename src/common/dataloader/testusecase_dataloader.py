from dataloader import DataLoader
# =================================
# This is an example of how to use the class dataloader
# on the server
# =================================

root_dir = '/Users/nils/Desktop/test'

args_dict = {'capture_dir': root_dir + '/annotations',
             'train_images_dir': root_dir + '/train2014',
             'val_images_dir': root_dir + '/val2014'}

batch_size = 2

dl = DataLoader(args_dict)
train_gen = dl.generator('train', batch_size)

for x in train_gen:
    print(x)
