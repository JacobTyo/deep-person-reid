from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import os
import shutil
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
        train = self.process_dir(self.train_dir, real_mil=self.real_mil)
        query = self.process_dir(self.query_dir, real_mil=False)
        gallery = self.process_dir(self.gallery_dir, real_mil=False)

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
        return text.replace('"', '').replace("'", '').replace(' ', '').replace('\n', '').replace('\r', '')

    def process_dir(self, dir_path, real_mil=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # save the image. Format: person(cluster/label)ID_detectedObjectID_eventID.png
        pattern = re.compile(r'([-\d]+)_[-\d]+_([-\d]+).png')
        object_id_to_image_id = {}

        if real_mil:
            # load the object_id_to_image_id.csv file, and label with respect to the image, not the object
            with open(os.path.join(self.dataset_dir, 'object_id_to_image_id.csv'), 'r') as f:
                for line_idx, line in enumerate(f.readlines()):
                    if line_idx == 0:
                        # skip headers
                        continue
                    object_id, image_id = line.strip().split(',')[:2]
                    object_id, image_id = self.clean_csv_map_data(object_id), self.clean_csv_map_data(image_id)

                    object_id_to_image_id[object_id] = image_id

        data = []
        for img_path in img_paths:
            person_id, event_id = map(int, pattern.search(img_path).groups())

            # make classes incrementing
            print(f'person_id: {person_id}')
            print(type(person_id))
            label_id = person_id if not real_mil else object_id_to_image_id[person_id]
            if label_id not in self.id_mapping:
                self.id_mapping[label_id] = self.id_counter
                self.id_counter += 1
            label = self.id_mapping[label_id]
            data.append((img_path, label, event_id))

        return data
