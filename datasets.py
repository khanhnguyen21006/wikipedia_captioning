import torch

import abc
import h5py
import json
import os
import re

from pycocotools.coco import COCO
from PIL import Image


def build_context(_batch, k):
    _bs = len(list(_batch.values())[0])
    l = k.split('_')
    if len(l) > 1:
        return [_batch[_l] if _l in _batch else [None]*_bs for _l in l]
    else:
        return _batch[k] if k in _batch else [None]*_bs

class BaseDataset(abc.ABC, torch.utils.data.Dataset):
    def __init__(self, d_folder, d_name, transform=None):
        self.d_folder = d_folder
        self.d_name = d_name
        self.transform = transform

        self.d_size = 0
        self.load_data()

    @abc.abstractmethod
    def load_data(self):
        pass

    def __len__(self):
        return self.d_size

    def collate(self, batch):
        # import pudb; pu.db
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        dict_batch['image'] = torch.stack(dict_batch['image'])
        for k in self.hparams['context_keys']:
            dict_batch[f'{k}_id'], dict_batch[f'{k}_mask'] = self.hparams['enc_tokenizer'].tokenize(
                    build_context(dict_batch, k), self.hparams['text_max_len']
                )
        for k in self.hparams['target_keys']:
            dict_batch[f'{k}_id'], dict_batch[f'{k}_mask'] = self.hparams['dec_tokenizer'].tokenize(
                    build_context(dict_batch, k), 100
                )
        return dict_batch

@BaseDataset.register
class WitDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'train', 'val', 'test', 'test_1k_RET', 'test_5k_RET'}

        super().__init__(*args, **kwargs)

    def load_data(self):
        self.images = None
        with h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_' + self.d_name + '.hdf5'), 'r') as h:
            self.d_size = len(h['images'])
        with open(os.path.join(self.d_folder, self.split + '_STRDESCS_' + self.d_name + '.json'), 'r') as f:
            self.descriptions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_STRCONTEXTS_' + self.d_name + '.json'), 'r') as f:
            self.contexts = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_STRCAPS_' + self.d_name + '.json'), 'r') as f:
            self.captions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_' + self.d_name + '.json'), 'r') as f:
            self.ids = json.load(f)

    def open_h5py(self):
        h = h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_' + self.d_name + '.hdf5'), 'r')
        self.images = h['images']

    def __getitem__(self, i):
        if self.images is None:
            self.open_h5py()

        image = self.images[i]
        description = self.descriptions[i]
        context = self.contexts[i]
        caption = self.captions[i]
        image_id = self.ids[i]

        if self.transform is not None:
            image = self.transform(image.transpose(1, 2, 0))

        ret = {
            'image': image,
            'image_id': image_id,
            'description': description,
            'section': context,
            'caption': caption
        }
        return ret

@BaseDataset.register
class CocoDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'train', 'val', 'test',}

        super().__init__(*args, **kwargs)

    def load_data(self):
        roots, all_anno_ids = self.get_coco_paths(self.d_folder)

        im_folder = roots[self.split]['img']
        cap_json = roots[self.split]['cap']
        ids = all_anno_ids[self.split]

        self.root_split = im_folder
        if isinstance(cap_json, tuple):
            self.coco_split = (COCO(cap_json[0]), COCO(cap_json[1]))
        else:
            self.coco_split = (COCO(cap_json),)
            self.root_split = (im_folder,)

        if isinstance(ids, tuple):
            self.bp = len(ids[0])
            self.ids = list(ids[0]) + list(ids[1])
        else:
            self.bp = len(ids)
            self.ids = ids

        self.d_size = len(self.ids)

    def __getitem__(self, i):
        if i < self.bp:
            coco, root = self.coco_split[0], self.root_split[0]
        else:
            coco, root = self.coco_split[1], self.root_split[1]
        ann_id = self.ids[i]
        caption = coco.anns[ann_id]['caption']
        image_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')  # transpose(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)
        cap_id, cap_len = self.tokenize(caption)

        ret = {
            'image': image,
            'image_id': image_id,
            'anno_id': ann_id,
            'caption': caption
        }
        return ret

    def get_coco_paths(self, path, use_restval=True):
        roots, ids = {}, {}
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json'),
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap']),
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']

        return roots, ids

@BaseDataset.register
class WitToyDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'train', 'val', 'test'}

        super().__init__(*args, **kwargs)  

    def load_data(self):
        with open(os.path.join(self.d_folder, self.split + '_SENTDESCS_' + self.d_name + '.json'), 'r') as f:
            self.descriptions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_SENTCONTEXTS_' + self.d_name + '.json'), 'r') as f:
            self.contexts = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_SENTCAPS_' + self.d_name + '.json'), 'r') as f:
            self.captions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_' + self.d_name + '.json'), 'r') as f:
            self.ids = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGENAMES_' + self.d_name + '.json'), 'r') as f:
            self.im_names = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEURLS_' + self.d_name + '.json'), 'r') as f:
            self.im_urls = json.load(f)

        self.d_size = len(self.ids)
        self.n_sample = 0

    def __getitem__(self, i):
        image_id = self.ids[i]
        im_name = self.im_names[i]
        im_url = self.im_urls[i]
        d_sents = self.descriptions[i]
        s_sents = self.contexts[i]
        c_sents = self.captions[i]

        image = Image.open(os.path.join(self.d_folder, 'images', im_name)).convert('RGB')  # transpose(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)

        description = merge(d_sents)
        context = merge(s_sents)
        caption = merge(c_sents)

        ret = {
            'image': image,
            'image_id': image_id,
            'image_url': im_url,
            'description': description,
            'section': context,
            'caption': caption
        }
        return ret

    def merge(sents):
        sampled = sents
        if self.n_sample > 0 and len(sents) > self.n_sample:
            idx = np.random.randint(len(sents) - self.n_sample)
            sampled = sents[idx:(idx+self.n_sample)]
        sent = ""
        for s in sampled:
            sent += ' ' + s
        sent = sent.strip()
        return sent

@BaseDataset.register
class GoodNewsDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'train', 'val', 'test'}

        super().__init__(*args, **kwargs)

    def load_data(self):
        self.images = None
        with h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_' + self.d_name + '.hdf5'), 'r') as h:
            self.d_size = len(h['images'])
        with open(os.path.join(self.d_folder, self.split + '_STRCONTEXTS_' + self.d_name + '.json'), 'r') as f:
            self.articles = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_STRCAPS_' + self.d_name + '.json'), 'r') as f:
            self.captions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_' + self.d_name + '.json'), 'r') as f:
            self.ids = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEURLS_' + self.d_name + '.json'), 'r') as f:
            self.im_urls = json.load(f)

    def open_h5py(self):
        h = h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_' + self.d_name + '.hdf5'), 'r')
        self.images = h['images']

    def __getitem__(self, i):
        if self.images is None:
            self.open_h5py()

        image = self.images[i]
        article = self.articles[i]
        caption = self.captions[i]
        image_id = self.ids[i]
        im_url = self.im_urls[i]

        if self.transform is not None:
            image = self.transform(image.transpose(1, 2, 0))

        ret = {
            'image': image,
            'image_id': image_id,
            'image_url': im_url,
            'section': article,
            'caption': caption
        }
        return ret
