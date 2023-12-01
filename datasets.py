import torch

import abc
import os, json, h5py, re
import numpy as np

from pycocotools.coco import COCO
from PIL import Image


def build_context(_batch, k):
    _bs = len(list(_batch.values())[0])
    l = k.split('_')
    if len(l) > 1:
        return [_batch[_l] if _l in _batch else [None]*_bs for _l in l]
    else:
        return _batch[k] if k in _batch else [None]*_bs


def merge(sents, n, do_sample=True):
    assert isinstance(sents, list)
    sampled = sents
    if do_sample:
        assert n > 0
        if len(sents) > n:
            idx = np.random.randint(len(sents) - n)
            sampled = sents[idx:(idx+n)]
    else:
        sampled = sents[:n]
    sent = ""
    for s in sampled:
        sent += ' ' + s
    sent = sent.strip()
    return sent

class BaseDataset(abc.ABC, torch.utils.data.Dataset):
    def __init__(self, d_folder, d_name, transform=None):
        self.d_folder = d_folder
        self.d_name = d_name
        self.transform = transform

        self.d_size = 0
        self.load_data()

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

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

    def load_extracted_context(self, metric):
        if not hasattr(self, 'context_sents') and not hasattr(self, 'description_sents'):
            if '_RET' not in self.split:
                if metric is not None:
                    with open(os.path.join(self.d_folder, f"extracted/{metric}", self.split + '_STRCONTEXTS_BYDESC_' + self.d_name + '.json'), 'r') as f:
                        self.extbydesc_contexts = json.load(f)
                    with open(os.path.join(self.d_folder, f"extracted/{metric}", self.split + '_STRCONTEXTS_BYCAP_' + self.d_name + '.json'), 'r') as f:
                        self.extbycap_contexts = json.load(f)
                with open(os.path.join(self.d_folder, 'extracted', self.split + '_STRCONTEXTS_SENTS_' + self.d_name + '.json'), 'r') as f:
                    self.context_sents = json.load(f)
                with open(os.path.join(self.d_folder, 'extracted', self.split + '_STRDESCS_SENTS_' + self.d_name + '.json'), 'r') as f:
                    self.description_sents = json.load(f)

    def __getitem__(self, i):
        if self.images is None:
            self.open_h5py()

        image = self.images[i]
        if self.hparams['extract_context'] != '':
            self.load_extracted_context(self.hparams['metric'])
            if re.compile(r"first_\d+").match(self.hparams['extract_context']):
                n = int(self.hparams['extract_context'].split('_')[-1])
                description = merge(self.description_sents[i], n, do_sample=False)
            elif re.compile(r"random_\d+").match(self.hparams['extract_context']):
                n = int(self.hparams['extract_context'].split('_')[-1])
                description = merge(self.description_sents[i], n)
        else:
            description = self.descriptions[i]

        if self.hparams['extract_context'] != '':
            self.load_extracted_context(self.hparams['metric'])
            if re.compile(r"by_desc_\d+").match(self.hparams['extract_context']):
                pad = int(self.hparams['extract_context'].split('_')[-1]) // 2
                sents = self.context_sents[i]
                ind = sents.index(self.extbydesc_contexts[i])
                context = ' '.join(sents[ind-pad:ind+pad+1]).strip()
            elif re.compile(r"by_cap_\d+").match(self.hparams['extract_context']):
                pad = int(self.hparams['extract_context'].split('_')[-1]) // 2
                sents = self.context_sents[i]
                ind = sents.index(self.extbycap_contexts[i])
                context = ' '.join(sents[ind-pad:ind+pad+1]).strip()
            elif re.compile(r"first_\d+").match(self.hparams['extract_context']):
                n = int(self.hparams['extract_context'].split('_')[-1])
                context = merge(self.context_sents[i], n, do_sample=False)
            elif re.compile(r"random_\d+").match(self.hparams['extract_context']):
                n = int(self.hparams['extract_context'].split('_')[-1])
                context = merge(self.context_sents[i], n)
        else:
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
        if self.hparams['wiki_context']:
            ret.update({
                'section': description + '. ' + context,
            })
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

        description = merge(d_sents, self.n_sample)
        context = merge(s_sents, self.n_sample)
        caption = merge(c_sents, self.n_sample)

        ret = {
            'image': image,
            'image_id': image_id,
            'image_url': im_url,
            'description': description,
            'section': context,
            'caption': caption
        }
        return ret

@BaseDataset.register
class WitPageDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'train', 'val', 'test', 'test_1k_RET', 'test_5k_RET'}

        super().__init__(*args, **kwargs)

    def load_data(self):
        self.images = None
        with h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_wit.hdf5'), 'r') as h:
            self.d_size = len(h['images'])
        with open(os.path.join(self.d_folder, self.split + '_STRDESCS_wit.json'), 'r') as f:
            self.descriptions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_STRCONTEXTS_wit.json'), 'r') as f:
            self.contexts = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_STRCAPS_wit.json'), 'r') as f:
            self.captions = json.load(f)
        with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_wit.json'), 'r') as f:
            self.ids = json.load(f)
        self.ret_mode = '_RET' in self.split
        if not self.ret_mode:
            with open(os.path.join(self.d_folder, 'page', self.split + '_IMAGEINPAGEIDS_wit.json'), 'r') as f:
                self.image_in_page = json.load(f)
            with open(os.path.join(self.d_folder, 'page', self.split + '_CONTEXTINPAGEIDS_wit.json'), 'r') as f:
                self.context_in_page = json.load(f)

    def open_h5py(self):
        h = h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_wit.hdf5'), 'r')
        self.images = h['images']

    def __getitem__(self, i):
        if self.images is None:
            self.open_h5py()

        image = self.images[i]
        description = self.descriptions[i]
        context = self.contexts[i]
        caption = self.captions[i]
        image_id = self.ids[i]

        ret = {
            'image_0': self.do_transform(image),
            'image_id': image_id,
            'description': description,
            'section_0': context,
            'caption': caption,
        }
        if not self.ret_mode:
            ret.update(self.extend_context(i))
        return ret

    def do_transform(self, image):
        if self.transform is not None:
            image = self.transform(image.transpose(1, 2, 0))
        return image

    def extend_context(self, i):
        ret = dict()
        n_space = self.hparams["num_space"]
        im_inpage = self.image_in_page[i]
        sec_inpage = self.context_in_page[i]
        if len(im_inpage) > (n_space - 1):
            im_inpage = np.random.choice(im_inpage, size=n_space-1, replace=False)
        else:
            while len(im_inpage) < (n_space - 1):
                im_inpage += [None]
        if len(sec_inpage) > (n_space - 1):
            sec_inpage = np.random.choice(sec_inpage, size=n_space-1, replace=False)
        else:
            while len(sec_inpage) < (n_space - 1):
                sec_inpage += [None]
        for _i in range(len(im_inpage)):
            im_i = None if im_inpage[_i] == None else self.do_transform(self.images[im_inpage[_i]])
            ret.update({f"image_{_i+1}": im_i})
        for _i in range(len(sec_inpage)):
            if isinstance(sec_inpage[_i], str):
                sec_i = None if sec_inpage[_i] == None else sec_inpage[_i]
            else:
                sec_i = None if sec_inpage[_i] == None else self.contexts[sec_inpage[_i]]
            ret.update({f"section_{_i+1}": sec_i})
        return ret

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        image_keys = sorted([key for key in keys if 'image' in key and key not in ['image_id', 'image_url']])
        section_keys = sorted([key for key in keys if 'section' in key])
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        dict_batch['image_ext_mask'] = torch.tensor([_im != None for _key in image_keys for _im in dict_batch[_key]])
        dict_batch['section_ext_mask'] = torch.tensor([_sec != None  for _key in section_keys for _sec in dict_batch[_key]])
        dict_batch['section'] = [_sec for _key in section_keys for _sec in dict_batch[_key] if _sec is not None]
        dict_batch['image'] = torch.stack([_im for _key in image_keys for _im in dict_batch[_key] if _im is not None])

        for k in self.hparams['context_keys']:
            dict_batch[f'{k}_id'], dict_batch[f'{k}_mask'] = self.hparams['enc_tokenizer'].tokenize(
                    build_context(dict_batch, k), self.hparams['text_max_len']
                )
        return dict_batch

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

class WitRetMultiDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self.split = split
        assert split in {'test_1k_RET', 'test_5k_RET', 'test_ImDeDup_RET', 
                            'test_ImSecDeDup_RET', 'test_CapDeDup_RET', 'test_SecDeDup_RET'} \
                            or re.match(r"test_(1k|5k)_RET_(Im|Sec)(_\d+)*", split)

        super().__init__(*args, **kwargs)

    def load_data(self):
        if 'ImDeDup' in self.split:
            self.images = None
            with h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_wit.hdf5'), 'r') as h:
                self.d_size = len(h['images'])
            with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_wit.json'), 'r') as f:
                self.ids = json.load(f)
            with open(os.path.join(self.d_folder, self.split + '_SecGTS_wit.json'), 'r') as f:
                self.gts = json.load(f)
        elif 'CapDeDup' in self.split:
            with open(os.path.join(self.d_folder, self.split + '_STRCAPS_wit.json'), 'r') as f:
                self.captions = json.load(f)
            with open(os.path.join(self.d_folder, self.split + '_ImGTS_wit.json'), 'r') as f:
                self.gts = json.load(f)
            self.d_size = len(self.captions)
        elif 'SecDeDup' in self.split:
            with open(os.path.join(self.d_folder, self.split + '_STRCONTEXTS_wit.json'), 'r') as f:
                self.contexts = json.load(f)
            with open(os.path.join(self.d_folder, self.split + '_ImGTS_wit.json'), 'r') as f:
                self.gts = json.load(f)
            self.d_size = len(self.contexts)
        elif '_RET' in self.split:  # get sentence tokenized data
            t = self.split.split('_')
            self.split, self.mod, self.ext, self.wd = '_'.join(t[:-3]), t[-3], t[-2], int(t[-1])
            if self.mod == 'Im':
                self.images = None
                with h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_wit.hdf5'), 'r') as h:
                    self.d_size = len(h['images'])
                with open(os.path.join(self.d_folder, self.split + '_IMAGEIDS_wit.json'), 'r') as f:
                    self.ids = json.load(f)
                self.gts = np.arange(self.d_size)[None, :].transpose(1, 0)
            elif self.mod == 'Sec':
                with open(os.path.join(self.d_folder, self.split + '_STRCONTEXTS_SENTS_wit.json'), 'r') as f:
                    self.context_sents = json.load(f)
                if self.ext == 'first':
                    self.gts = [_i for _i in range(len(self.context_sents))]
                    self.contexts = [_c[0] if len(_c) > 0 else '' for _c in self.context_sents]
                else:
                    self.gts = []
                    for ind, c in enumerate(self.context_sents):
                        if not len(c) < self.wd:
                            for _ in range(len(c) - self.wd + 1):
                                self.gts.append(ind)
                        else:
                            self.gts.append(ind)
                    self.contexts = []
                    for c in self.context_sents:
                        if not len(c) < self.wd:
                            for _i in range(len(c) - self.wd + 1):
                                self.contexts.append(' '.join(c[_i:_i + self.wd]).strip())
                        else:
                            self.contexts.append(' '.join(c).strip())
                self.d_size = len(self.contexts)
                assert len(self.contexts) == len(self.gts)
            else:
                raise ValueError("Wrong Mod")
        else:
            raise ValueError("Invalid split.")

    def open_h5py(self):
        h = h5py.File(os.path.join(self.d_folder, self.split + '_IMAGES_wit.hdf5'), 'r')
        self.images = h['images']

    def __getitem__(self, i):
        if 'ImDeDup' in self.split:
            if self.images is None:
                self.open_h5py()
            image = self.images[i]
            image_id = self.ids[i]
            if self.transform is not None:
                image = self.transform(image.transpose(1, 2, 0))
            ret = {'image': image, 'image_id': image_id,}
        elif 'CapDeDup' in self.split:
            caption = self.captions[i]
            ret = {'caption': caption}
        elif 'SecDeDup' in self.split:
            context = self.contexts[i]
            ret = {'section': context}
        elif '_RET' in self.split:  # get sentence tokenized data
            if self.mod == 'Im':
                if self.images is None:
                    self.open_h5py()
                image = self.images[i]
                image_id = self.ids[i]
                if self.transform is not None:
                    image = self.transform(image.transpose(1, 2, 0))
                ret = {'image': image, 'image_id': image_id,}
            elif self.mod == 'Sec':
                context = self.contexts[i]
                ret = {'section': context}
            else:
                raise ValueError("Wrong modality")
        else:
            raise ValueError("Invalid split")
        ret.update({'retrieve_gt': self.gts[i]})
        return ret

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        if 'ImDeDup' in self.split or 'ImSecDeDup' in self.split or ('_RET' in self.split and self.mod == 'Im'):
            dict_batch['image'] = torch.stack(dict_batch['image'])
        if 'CapDeDup' in self.split:
            dict_batch['caption_id'], dict_batch['caption_id'] = self.hparams['enc_tokenizer'].tokenize(
                    dict_batch['caption'], self.hparams['text_max_len']
                )
        if 'SecDeDup' in self.split or ('_RET' in self.split and self.mod == 'Sec'):
            dict_batch['section_id'], dict_batch['section_mask'] = self.hparams['enc_tokenizer'].tokenize(
                    dict_batch['section'], self.hparams['text_max_len']
                )
        return dict_batch

class RLWitDataset(WitDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(split, *args, **kwargs)

    def load_data(self):
        self.d_name = 'wit'
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

    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [b[k] if k in b else None for b in batch] for k in keys}

        dict_batch['image'] = torch.stack(dict_batch['image'])
        for k in ['section', 'description']:
            dict_batch[f'{k}_id'], dict_batch[f'{k}_mask'] = self.hparams['dec_tokenizer'].tokenize(
                    build_context(dict_batch, k), self.hparams['text_max_len']
                )
        for k in ['caption']:
            dict_batch[f'{k}_id'], dict_batch[f'{k}_mask'] = self.hparams['dec_tokenizer'].tokenize(
                    build_context(dict_batch, k), 100
                )
        return dict_batch