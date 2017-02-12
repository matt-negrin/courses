from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras.utils.data_utils import get_file
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
import threading

def mn_valid(validation_frac=.3, local=False):
    if local:
        path = '/Users/matthew.negrin/deeplearning/courses/deeplearning1/nbs/data/statefarm/'
    else:
        path = '/home/ubuntu/matt/courses/deeplearning1/nbs/data/statefarm/'
    direct_list = ['valid/'+x[0].split('/')[-1] for x in os.walk(path+'train/') if x[0].split('/')[-1] != '']
    direct_list.insert(0, "valid")
    for directory in direct_list:
        if not os.path.exists(path+directory):
            os.makedirs(path+directory)
    dil = pd.read_csv(path + 'driver_imgs_list.csv')
    valid_subjects = dil.groupby(['subject']).size().sample(frac=validation_frac).keys().tolist()
    valid_frame = dil[dil['subject'].isin(valid_subjects)]
    for (subject, classname, img) in valid_frame.values:
        source = '{}train/{}/{}'.format(path, classname, img)
        target = source.replace('train', 'valid')
        print('mv {} {}'.format(source, target))
        os.rename(source, target)

def rm_sample_set(local=False):
    if local:
        path = '/Users/matthew.negrin/deeplearning/courses/deeplearning1/nbs/data/statefarm/'
    else:
        path = '/home/ubuntu/matt/courses/deeplearning1/nbs/data/statefarm/'
    rmtree(path+'sample/')
        
def mn_sample(local=False):
    if local:
        path = '/Users/matthew.negrin/deeplearning/courses/deeplearning1/nbs/data/statefarm/'
    else:
        path = '/home/ubuntu/matt/courses/deeplearning1/nbs/data/statefarm/'
    os.makedirs(path+'sample')
    os.makedirs(path+'sample/train')
    os.makedirs(path+'sample/valid')
    os.makedirs(path+'sample/models')
    os.makedirs(path+'sample/results')
    direct_list = [x[0].split('/')[-1] for x in os.walk(path+'train/') if x[0].split('/')[-1] != '']
    for d in direct_list:
        os.makedirs(path+'sample/train/'+d)
        os.mkdir(path+'sample/valid/'+d)
    tg = glob(path+'train/c?/*.jpg')
    shuf = np.random.permutation(tg)
    for i in range(1500): copyfile(shuf[i],path+'sample/train/' + shuf[i].split('/')[-2]+'/'+shuf[i].split('/')[-1])
    vg = glob(path+'valid/c?/*.jpg')
    vshuf = np.random.permutation(vg)
    for i in range(1000): copyfile(vshuf[i],path+'sample/valid/' + vshuf[i].split('/')[-2]+'/'+vshuf[i].split('/')[-1])    

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16():
    """The VGG 16 Imagenet model"""


    def __init__(self):
        self.FILE_PATH = 'http://www.platform.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def create(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        model = self.model
        last_conv_idx = [index for index,layer in enumerate(model.layers) if type(layer) is Convolution2D][-1]
        fc_layers = model.layers[last_conv_idx+1:]
        for layer in fc_layers: model.pop()
        for layer in model.layers: layer.trainable=False
        self.compile()

    def get_fc_model(self, p, batches):
        model = self.model
        return_model = Sequential([
            MaxPooling2D(input_shape=model.layers[-1].output_shape[1:]),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(p),
            BatchNormalization(axis=1),
            Dense(256, activation='relu'),
            Dropout(p),
            BatchNormalization(axis=1),
            Dense(batches.nb_class, activation='softmax')
            ])
        return return_model

    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

    def layer_divider(self):
        last_conv_idx = [index for index,layer in enumerate(self.model.layers) if type(layer) is Convolution2D][-1]
        conv_layers = self.model.layers[:last_conv_idx+1]
        fc_layers = self.model.layers[last_conv_idx+1:]
        return conv_layers, fc_layers   

class BcolzArrayIterator(object):
    def __init__(self, X, y=None, w=None, batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) '
                             'should have the same length. '
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')
        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        if y is not None:
            self.y = y[:]
        else:
            self.y = None
        if w is not None:
            self.w = w[:]
        else:
            self.w = None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed
        
    def reset(self):
        self.batch_index = 0
        
    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                if self.shuffle:
                    self.index_array = np.random.permutation(self.X.nchunks + 1)
                else:
                    self.index_array = np.arange(self.X.nchunks + 1)

            batches_x = []
            batches_y = []
            batches_w = []
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                self.total_batches_seen += 1
                if not self.y is None:
                    batches_y.append(self.y[current_index * self.X.chunklen: current_index * self.X.chunklen + current_batch_size])
                if not self.w is None:
                    batches_w.append(self.w[current_index * self.X.chunklen: current_index * self.X.chunklen + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break
            
        batch_x = np.concatenate(batches_x)
        if self.y is None:
            return batch_x
        
        batch_y = np.concatenate(batches_y)
        if self.w is None:
            return batch_x, batch_y
        
        batch_w = np.concatenate(batches_w)
        return batch_x, batch_y, batch_w

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)