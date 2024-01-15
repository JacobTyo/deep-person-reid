from __future__ import division, print_function, absolute_import
import copy
import glob
import os.path as osp
from tqdm import tqdm

from ..dataset import ImageDataset


class SYSU30k(ImageDataset):
    """SYSU-30k.

    This is a weakly supervised dataset for person search. The labels are bag level labels.

    Dataset format:
    SYSU-30k-released
    ├── SYSU-30k-released
    │   ├── meta
    │   |   ├── train.txt (for weakly supervised training, "filename\n" in each line)
    │   |   ├── query.txt (for evaluation)
    │   |   ├── gallery.txt (for evaluation)
    │   ├── sysu_train_set_all
    │   |   ├── 0000000001
    │   |   ├── 0000000002
    │   |   ├── 0000000003
    │   |   ├── 0000000004
    │   |   ├── ...
    │   |   ├── 0000028309
    │   |   ├── 0000028310
    │   ├── sysu_test_set_all
    │   |   ├── gallery
    │   |   |   ├── 000028311
    │   |   |   |   ├── 000028311_c1_1.jpg
    │   |   |   ├── 000028312
    │   |   |   |   ├── 000028312_c1_1.jpg
    │   |   |   ├── 000028313Leaderboard on SYSU-30k.
    │   |   |   |   ├── 000028313_c1_1.jpg
    │   |   |   ├── 000028314
    │   |   |   |   ├── 000028314_c1_1.jpg
    │   |   |   ├── ...
    │   |   |   |   ├── ...
    │   |   |   ├── 000029309
    │   |   |   |   ├── 000029309_c1_1.jpg
    │   |   |   ├── 000029310
    │   |   |   |   ├── 000029310_c1_1.jpg
    │   |   |   ├── 0000others
    │   |   |   |   ├── 0000others_c1_1.jpg
    │   |   |   |   ├── ...
    │   |   |   |   ├── ...
    │   |   ├── query
    │   |   |   ├── 000028311
    │   |   |   |   ├── 000028311_c2_2.jpg
    │   |   |   ├── 000028312
    │   |   |   |   ├── 000028312_c2_2.jpg
    │   |   |   ├── 000028313
    │   |   |   |   ├── 000028313_c2_2.jpg
    │   |   |   ├── 000028314
    │   |   |   |   ├── 000028314_c2_2.jpg
    │   |   |   ├── ...
    │   |   |   |   ├── ...
    │   |   |   ├── 000029309
    │   |   |   |   ├── 000029309_c2_2.jpg
    │   |   |   ├── 000029310
    │   |   |   |   ├── 000029310_c2_2.jpg


    Reference:
        Wang et al. Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark.

    URL: `<https://github.com/wanggrun/SYSU-30k>`_

    Dataset statistics:
        - identities: 30,508
        - images: 29,606,918
    """
    _dataset_dir = 'SYSU-30k-released'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self._dataset_dir)

        self.train_files_list = osp.join(self.dataset_dir, 'meta', 'train.txt')
        self.gallery_files_list = osp.join(self.dataset_dir, 'meta', 'gallery.txt')
        self.query_files_list = osp.join(self.dataset_dir, 'meta', 'query.txt')

        self.train_files_dir = osp.join(self.dataset_dir, 'sysu_train_set_all')
        self.gallery_files_dir = osp.join(self.dataset_dir, 'sysu_test_set_all', 'gallery')
        self.query_files_dir = osp.join(self.dataset_dir, 'sysu_test_set_all', 'query')

        # keep the pid2label conversion consistent across sets.
        self.pid2label = {}
        self.pid2label['0000others'] = -1
        self.cam2camid = {}

        # image name format:
        train = self.process_txt_file(self.train_files_dir, self.train_files_list)
        query = self.process_txt_file(self.query_files_dir, self.query_files_list)
        gallery = self.process_txt_file(self.gallery_files_dir, self.gallery_files_list)

        super(SYSU30k, self).__init__(train, query, gallery, **kwargs)

    def process_txt_file(self, base_dir, txt_file):
        """
        Return a list of tuples (img_path, pid, camid). Just use 0 for the camid as this dataset doesn't have them.
        The pid is the folder name (see classdescription)

        Args:
            base_dir: where the files are located
            txt_file: a list of the files of interest

        Returns:
            A list of tuples (img_path, pid, camid)
        """
        with open(txt_file, 'r') as f:
            img_paths = f.readlines()
        num_imgs = len(img_paths)
        assert num_imgs > 0, "No images found in {}".format(txt_file)
        print("=> Found {} images in {}.".format(num_imgs, txt_file))
        print("=> Loading image paths into (x,y) ...")

        # get all identities:
        # pid_container = set()
        # for img_path in tqdm(img_paths):
        #     # if the image path doesn't exist, don't include it
        #     if not osp.exists(osp.join(base_dir, img_path.strip())):
        #         continue
        #     pid = img_path.split('/')[0].strip()
        #     pid_container.add(pid)
        #     # if 'train' in txt_file and len(pid_container) >= 1000:
        #     #     break
        # for pid in pid_container:
        #     if pid not in self.pid2label:
        #         self.pid2label[pid] = len(self.pid2label)

        # num_pids = len(self.pid2label)
        # assert num_pids > 0, "No identities found in {}".format(txt_file)
        # print("=> Found {} identities in {}.".format(num_pids, txt_file))

        # extract data
        data = []
        for img_path in img_paths:
            pid = img_path.split('/')[0].strip()
            if pid not in self.pid2label:
                self.pid2label[pid] = len(self.pid2label)
            label = self.pid2label[pid]
            # get the camera id, if it exists:
            try:
                fname = img_path.split('/')[-1]
                camera = fname.split('c')[1].strip()
            except:
                camera = 0
            if camera not in self.cam2camid:
                self.cam2camid[camera] = len(self.cam2camid)
            data.append((osp.join(base_dir, img_path.strip()), label, self.cam2camid[camera]))
        return data
