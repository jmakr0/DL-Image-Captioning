import numpy as np
import os
from args import get_parser
from utils.dataloader import DataLoader
from utils.config import get_opt
from utils.lang_proc import idx2word, sample, beamsearch
from model import get_model, image_model, language_model
from keras.models import Model
import pickle
import json
import time
from keras import backend as K
from keras.layers import Input
import json

parser = get_parser()
args_dict = parser.parse_args()
args_dict.mode = 'test'
args_dict.bs = 1

model = get_model(args_dict)
opt = get_opt(args_dict)
weights = args_dict.model_file
model.load_weights(weights)
model.compile(optimizer=opt,loss='categorical_crossentropy')

# load vocab to convert captions to words and compute cider
data = json.load(open(os.path.join(args_dict.data_folder,'data',args_dict.json_file),'r'))
vocab_src = data['ix_to_word']
inv_vocab = {}
for idx in vocab.keys():
    inv_vocab[int(idx)] = vocab_src[idx]
vocab = {v:k for k,v in inv_vocab.items()}

dataloader = DataLoader(args_dict)
N_train, N_val, N_test, _ = dataloader.get_dataset_size()
N = args_dict.bs
gen = dataloader.generator('test',batch_size=args_dict.bs,train_flag=False)
captions = []
num_samples = 0
print_every = 100
t = time.time()
for [ims,prevs],caps,_,imids in gen:

    if args_dict.bsize > 1:

        # beam search
        word_idxs = np.zeros((args_dict.bsize,args_dict.seqlen))
        word_idxs[:,:] = 2
        ### beam search caps ###
        conv_feats = cnn.predict_on_batch(ims)
        seqs,scores = beamsearch(model=lang_model,image=conv_feats,
                                 vocab_size = args_dict.vocab_size,
                                 start=0,eos=0,maxsample=args_dict.seqlen,
                                 k=args_dict.bsize)

        seqs = np.array(seqs)[np.argsort(scores)[::-1][:args_dict.bsize]]
        for i,seq in enumerate(seqs):
            word_idxs[i,:len(seq)-1] = seq[1:] # start token

    else:
        # greedy caps
        prevs = np.zeros((N,1))
        word_idxs = np.zeros((N,args_dict.seqlen))

        for i in range(args_dict.seqlen):
            # get predictions
            preds = model.predict_on_batch([ims,prevs]) #(N,1,vocab_size)
            preds = preds.squeeze()
            if args_dict.temperature > 0:
                preds = sample(preds,temperature=args_dict.temperature)

            word_idxs[:,i] = np.argmax(preds,axis=-1)
            prevs = np.argmax(preds,axis=-1)
            prevs = np.reshape(prevs,(N,1))

    pred_caps = idx2word(word_idxs,inv_vocab)
    true_caps = idx2word(np.argmax(caps,axis=-1),inv_vocab)
    pred_cap = ' '.join(pred_caps[0][:-1])# exclude eos
    true_cap = ' '.join(true_caps[0][:-1])# exclude eos

    captions.append({"image_id":imids[0]['id'],
                    "caption": pred_cap})
    num_samples+=1

    if num_samples%print_every==0:
        print ("%d/%d"%(num_samples,N_test))

    model.reset_states()
    if num_samples == N_test:
        break
print "Processed %s captions in %f seconds."%(len(captions),time.time() - t)
results_file = os.path.join(args_dict.data_folder, 'results',
                          args_dict.model_name +'_gencaps.json')
with open(results_file, 'w') as outfile:
    json.dump(captions, outfile)
print "Saved results in", results_file
