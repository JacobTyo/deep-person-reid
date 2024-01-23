from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import os
import shutil
from collections import defaultdict

from tqdm import tqdm
from ..dataset import ImageDataset


class PerformancePhoto(ImageDataset):
    """performancePhoto. """

    _junk_pids = [0, -1]
    dataset_dir = osp.join('reid-data', 'performance')
    endpoint = ''
    username = ''
    password = ''
    database_name = ''
    aws_access_key_id = ''
    aws_secret_access_key = ''
    region_name = ''
    s3_bucket = ''

    def __init__(self,
                 query_set='query_all',
                 gallery_set='gallery_all',
                 real_mil=False,
                 endpoint='',
                 username='',
                 password='',
                 database_name='',
                 aws_access_key_id='',
                 aws_secret_access_key='',
                 region_name='',
                 s3_bucket='',
                 **kwargs):
        self.root = osp.curdir
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_perf_dataset(
            self.dataset_dir,
            endpoint,
            username,
            password,
            database_name,
            aws_access_key_id,
            aws_secret_access_key,
            region_name,
            s3_bucket
        )

        self.id_mapping = {}
        self.photo_mapping = {}
        self.id_counter = 0

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.data_dir, 'query')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.query_dir = osp.join(self.data_dir, query_set)
        self.gallery_dir = osp.join(self.data_dir, gallery_set)
        self.real_mil = real_mil

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        # This needs fixed somehow - the query and test dirs are identical. That should not be the case.
        #   Query should be one this, and gallery another
        mil_extra_data_path = osp.join(self.data_dir, 'mil_images') if self.real_mil else None
        train = self.process_dir(self.train_dir, mil_extra_data_path=mil_extra_data_path)
        query = self.process_dir(self.query_dir, mil_extra_data_path=None)
        gallery = self.process_dir(self.gallery_dir, mil_extra_data_path=None)

        # can make this hard somehow? Different test sets? What about the eval I've done?

        super(PerformancePhoto, self).__init__(train, query, gallery, **kwargs)


    def download_perf_dataset(self,
                              dataset_dir,
                              endpoint,
                              username,
                              password,
                              database_name,
                              aws_access_key_id,
                              aws_secret_access_key,
                              region_name,
                              s3_bucket):

        print(f'dataset dir: {dataset_dir}')
        assert osp.exists(dataset_dir), 'dataset dir does not exist and must be manually prepared'
        return

    def clean_csv_map_data(self, text):
        return int(re.findall(r'\d+', text)[0])

    def process_dir_MIL(self, dir_path, mil_extra_data_path=None):
        print("Loading dataset for MIL learning.")
        # first build the relationships we'll need to make the dataset
        # each object only has one image
        objectid2imageid = defaultdict(int)
        # but each image can have multiple object
        imageid2objectid = defaultdict(set)
        # load the object_id_to_image_id.csv file
        with open(os.path.join(self.dataset_dir, 'object_id_to_image_id.csv'), 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                if line_idx == 0:
                    # skip headers
                    continue
                object_id, image_id = line.strip().split(',')[:2]
                object_id, image_id = self.clean_csv_map_data(object_id), self.clean_csv_map_data(image_id)

                objectid2imageid[object_id] = image_id
                imageid2objectid[image_id].add(object_id)

        # Now, interpret the person_id as a bag. Use this to build a dict of bag_ids to a list of image_ids
        bagids2objectids = defaultdict(set)
        objectids2bagids = defaultdict(set)
        objectid2imgpath = defaultdict(str)
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)_([-\d]+).png')
        for img_path in img_paths:
            bag_id, obj_id, event_id = map(int, pattern.search(img_path).groups())
            bagids2objectids[bag_id].add(obj_id)
            objectids2bagids[obj_id].add(bag_id)
            objectid2imgpath[obj_id] = img_path

        # now we need to construct a mapping that identifies what images comprise each bag.
        # we do this by taking each object_id in a bag, and then getting their image id
        bagids2imageids = defaultdict(set)
        imageids2bagids = defaultdict(set)
        for bag_id, object_ids in bagids2objectids.items():
            for object_id in object_ids:
                this_object_image_id = objectid2imageid[object_id]
                bagids2imageids[bag_id].add(this_object_image_id)
                imageids2bagids[this_object_image_id].add(bag_id)

        # and now finally, we can construct each bag fully, based off of all objects in each image
        data = []
        bag_id_map = {}
        for bag_id, image_ids in bagids2imageids.items():
            for image_id in image_ids:
                object_ids = imageid2objectid[image_id]
                for object_id in object_ids:
                    # needs to be image_path, bag_id (label), 0
                    if bag_id not in bag_id_map:
                        bag_id_map[bag_id] = len(bag_id_map)
                    label = bag_id_map[bag_id]
                    if object_id in objectid2imgpath:
                        # Many id's won't be in this dict, because they are in the other folder path
                        data.append((objectid2imgpath[object_id], label, 0))

        # great, now the normal data is dealt with, but we do not have the extra MIL data.
        img_paths = glob.glob(osp.join(mil_extra_data_path, '*.png'))
        pattern = re.compile(r'([-\d]+).png')
        for img_path in img_paths:
            obj_id = map(int, pattern.search(img_path).groups())
            # now add it to the dataset properly
            # first, get the image_id of the object
            image_id = objectid2imageid[obj_id]
            # now get the bag id's from the image id
            bag_id = imageids2bagids[image_id]
            for bid in bag_id:
                # if bid not in bag_id_map:
                #     bag_id_map[bid] = len(bag_id_map)
                # bag_id = bid
                # finally, the label
                label = bag_id_map[bid]
                data.append((img_path, label, 0))

        return data

    def process_dir(self, dir_path, mil_extra_data_path=None):
        if mil_extra_data_path:
            return self.process_dir_MIL(dir_path, mil_extra_data_path)

        # path to the well labeled re-id images
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # a regex pattern to extract information of interest
        pattern = re.compile(r'([-\d]+)_[-\d]+_([-\d]+).png')

        data = []

        for img_path in img_paths:
            person_id, event_id = map(int, pattern.search(img_path).groups())
            if person_id not in self.id_mapping:
                self.id_mapping[person_id] = self.id_counter
                self.id_counter += 1
            label = self.id_mapping[person_id]
            data.append((img_path, label, event_id))

        return data
