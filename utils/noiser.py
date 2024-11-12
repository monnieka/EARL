import numpy as np
from numpy.testing import assert_array_almost_equal
import random
import pickle
import os
from utils.conf import targets_path as path

def add_symmetric_noise(source_class, dataset):
    if dataset.train:
        filepath = os.path.join(path(), dataset.root.split('/')[-1], dataset.noise ,str(int(dataset.noise_rate*100)), 'noisy_targets')

        if os.path.exists(filepath):
            with open(filepath, 'rb') as infile:
                dataset.noisy_targets = pickle.load(infile)
            print('Noisy sym targets loaded from file!')            
            return
        
    for y in source_class:
        random_target = [t for t in source_class]
        random_target.remove(y)
        tindx = [i for i, x in enumerate(dataset.targets) if x == y] 
        for i in tindx[:round(len(tindx)*dataset.noise_rate)]: 
            dataset.noisy_targets[i] = random.choice(random_target)
    

def add_asymmetric_noise(source_class, target_class, dataset):
    # tmp = source_class.copy()
    # source_class += target_class
    # target_class += tmp
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(dataset.targets) == s)[0]
        n_noisy = int(dataset.noise_rate * cls_idx.shape[0]) 
        noisy_sample_index = np.random.choice(list(cls_idx), n_noisy, replace=False)
        for idx in noisy_sample_index:
            dataset.noisy_targets[idx] = t

def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.- symmetric setting 
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    
    for i in np.arange(size - 1):
        P[i, i+1] = noise      

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def noisify_cifar10_asymmetric(noise, dataset):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    if dataset.train:
        filepath = os.path.join(path(), dataset.root.split('/')[-1], dataset.noise ,str(int(dataset.noise_rate*100)), 'noisy_targets')

        if os.path.exists(filepath):
            with open(filepath, 'rb') as infile:
                dataset.noisy_targets = pickle.load(infile)
            print('Noisy asym targets loaded from file!')            
            return
        
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:

        P = (1. - noise) * np.eye(nb_classes)
        
        for i in np.arange(nb_classes - 1):
            P[i, i+1] = noise      

        # adjust last row
        P[nb_classes-1, 0] = noise            

        y_train_noisy = multiclass_noisify(dataset.targets, P=P)
        actual_noise = (y_train_noisy != dataset.targets).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
        
        dataset.noisy_targets = y_train_noisy.tolist()


def noisify_cifar100_asymmetric(noise, dataset):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    if dataset.train:
        filepath = os.path.join(path(), dataset.root.split('/')[-1], dataset.noise ,str(int(dataset.noise_rate*100)), 'noisy_targets')

        if os.path.exists(filepath):
            with open(filepath, 'rb') as infile:
                dataset.noisy_targets = pickle.load(infile)
            print('Noisy asym targets loaded from file!')            
            return
        
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(dataset.targets, P=P)
        actual_noise = (y_train_noisy != dataset.targets).mean()
        assert actual_noise > 0.0
    
        dataset.noisy_targets = y_train_noisy.tolist()

def multiclass_noisify(y, P):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    y = np.array(y)
    m = np.shape(y)[0]
    new_y = y.copy()
    flipper = np.random.RandomState(0)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y