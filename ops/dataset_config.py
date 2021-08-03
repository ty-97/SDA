# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET = "/data1/vip/Datasets/"



def return_diving48(modality):
    filename_categories = 48
    # filename_categories = 'soccer_allframe256_v2j/soccer_classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'diving48/'
        filename_imglist_train = 'trainValTest/train.txt'
        filename_imglist_val = 'trainValTest/val.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_epic55verb(modality):
    filename_categories = 125
    if modality == 'RGB':
        root_data = ROOT_DATASET +'EPIC_KITCHENS/'
        filename_imglist_train = root_data + 'trainValTest/train_55_verb.txt'
        filename_imglist_val = root_data + 'trainValTest/val_55_verb.txt'
        prefix = 'frame_{:010d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_epic55noun(modality):
    filename_categories = 352
    if modality == 'RGB':
        root_data = ROOT_DATASET +'EPIC_KITCHENS/'
        filename_imglist_train = root_data + 'trainValTest/train_55_noun.txt'
        filename_imglist_val = root_data + 'trainValTest/val_55_noun.txt'
        prefix = 'frame_{:010d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_somethingv1(modality):
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = root_data + 'some_some_v1/trainValTest/train.txt'
        filename_imglist_val = root_data + 'some_some_v1/trainValTest/val.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 174
    if modality == 'RGB':
        root_data = ROOT_DATASET
        filename_imglist_train = root_data + 'some_some_v2/trainValTest/train.txt'
        filename_imglist_val = root_data + 'some_some_v2/trainValTest/val.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_dataset(dataset, modality):
    dict_single = {'somethingv1': return_somethingv1, 'somethingv2': return_somethingv2,
                   'epic55verb': return_epic55verb, 'epic55noun': return_epic55noun,
                   'diving48': return_diving48, 
                   }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)
    
    categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
