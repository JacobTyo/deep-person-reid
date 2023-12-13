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
        self.id_counter = 0

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.data_dir, 'query')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.query_dir = osp.join(self.data_dir, query_set)
        self.gallery_dir = osp.join(self.data_dir, gallery_set)

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        # This needs fixed somehow - the query and test dirs are identical. That should not be the case.
        #   Query should be one this, and gallery another
        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

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
        # print('downloading dataset')
        #
        # s3 = boto3.client("s3",
        #                   aws_access_key_id=aws_access_key_id,
        #                   aws_secret_access_key=aws_secret_access_key,
        #                   region_name=region_name)
        #
        # sql = """SELECT do.cluster_id,
        #                 do.id,
        #                 pa.event_id
        #             FROM detected_objects do
        #             JOIN images i ON do.image_id = i.id
        #             JOIN photographer_albums pa ON i.photographer_album_id = pa.id
        #             WHERE do.cluster_id IS NOT NULL
        #        """
        #
        # with pymysql.connect(host=endpoint, user=username, password=password, database=database_name) as connection:
        #     with connection.cursor() as cursor:
        #         cursor.execute(sql)
        #         rows = cursor.fetchall()
        #
        # cluster_ids = [row[0] for row in rows]
        # unique_cluster_ids = list(set(cluster_ids))
        # # calculate index for 20% split
        # split_index = int(len(unique_cluster_ids) * 0.2)
        # cluster_ids_20 = unique_cluster_ids[:split_index]
        #
        # os.makedirs(dataset_dir, exist_ok=True)
        # os.makedirs(os.path.join(dataset_dir, 'bounding_box_train'), exist_ok=True)
        # os.makedirs(os.path.join(dataset_dir, 'bounding_box_test'), exist_ok=True)
        # os.makedirs(os.path.join(dataset_dir, 'query'), exist_ok=True)
        #
        # inc_camera_id = 0
        # for row in tqdm(rows):
        #     # get the image
        #     response = s3.get_object(Bucket=s3_bucket, Key=f'{row[1]}.png')['Body']
        #     # save the image. Format: person(cluster/label)ID_detectedObjectID_eventID.png
        #     # this has to be updated such that eventID is actually a cameraID
        #     if row[0] in cluster_ids_20:
        #         with open(os.path.join(dataset_dir, 'query', f'{row[0]}_{row[1]}_{inc_camera_id}.png'), 'wb') as out_file:
        #             shutil.copyfileobj(response, out_file)
        #         inc_camera_id += 1
        #     else:
        #         with open(os.path.join(dataset_dir, 'bounding_box_train', f'{row[0]}_{row[1]}_{inc_camera_id}.png'), 'wb') as out_file:
        #             shutil.copyfileobj(response, out_file)
        #         inc_camera_id += 1
        #
        # files_to_copy = os.listdir(os.path.join(dataset_dir, 'query'))
        # for file in files_to_copy:
        #     shutil.copy(os.path.join(dataset_dir, 'query', file), os.path.join(dataset_dir, 'bounding_box_test', file))


    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # save the image. Format: person(cluster/label)ID_detectedObjectID_eventID.png
        pattern = re.compile(r'([-\d]+)_[-\d]+_([-\d]+).png')

        data = []
        for img_path in img_paths:
            person_id, event_id = map(int, pattern.search(img_path).groups())
            if person_id not in self.id_mapping:
                self.id_mapping[person_id] = self.id_counter
                self.id_counter += 1
            data.append((img_path, self.id_mapping[person_id], event_id))
        return data
