import requests
from tqdm import tqdm
import os
import zipfile
import shutil

# link of dataset source https://utdallas.app.box.com/s/r5sx2ghgn62bg1tgjp9ily6jx2fifahl
# dataset image include:
# data1: 46.6 GB
# data2: 46.6 GB
# data3: 33.3 GB
# YCB Video Base: 373.2

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

def extract_zip(file_path, extract_to, target_dataset_root):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


        source_data_path = os.path.join(extract_to, 'data')

        # create terget directory of target dataset root
        os.makedirs(target_dataset_root, exist_ok=True)

        # delete zip file after extraction
        os.remove(file_path)

        # move all data from source_data_path to target_dataset_root

        for item in os.listdir(source_data_path):
            src_item = os.path.join(source_data_path, item)
            dst_item = os.path.join(target_dataset_root, item)

            # delete before move if dst_item exists
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    os.rmdir(dst_item)
                else:
                    os.remove(dst_item)
            
            shutil.move(src_item, dst_item)

if __name__ == "__main__":

    data1_link = ""
    data2_link = ""
    data3_link = ""
    ycb_video_base_link = ""

    data_path = './dataset/data'

    os.makedirs(data_path, exist_ok=True)

    # download dataset zip files
    download_file(data1_link, os.path.join(data_path, 'data1.zip'))
    download_file(data2_link, os.path.join(data_path, 'data2.zip'))
    download_file(data3_link, os.path.join(data_path, 'data3.zip'))
    download_file(ycb_video_base_link, os.path.join(data_path, 'ycb_video_base.zip'))

    # extract dataset zip files
    extract_zip(os.path.join(data_path, 'data1.zip'), data_path, os.path.join(data_path, 'data1'))
    extract_zip(os.path.join(data_path, 'data2.zip'), data_path, os.path.join(data_path, 'data2'))
    extract_zip(os.path.join(data_path, 'data3.zip'), data_path, os.path.join(data_path, 'data3'))
    extract_zip(os.path.join(data_path, 'ycb_video_base.zip'), data_path, os.path.join(data_path, 'ycb_video_base'))
