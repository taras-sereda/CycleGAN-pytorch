import os
import shutil
import argparse
import random
import datetime as dt

import skimage
import skimage.io as io

parser = argparse.ArgumentParser('script for splitting AMOS dataset '
                                 'for day and night images')
parser.add_argument('--data_root', default='/Users/taras/datasets/horse2zebra/testA', type=str)
parser.add_argument('--dest_root', default='./', type=str)
parser.add_argument('--day_threshold', default=0.3, type=float)
parser.add_argument('--folder_num_imgs', default=10, type=int)

args = parser.parse_args()

day_root = os.path.join(args.dest_root, 'day')
night_root = os.path.join(args.dest_root, 'night')

if not os.path.exists(day_root):
    os.makedirs(day_root)
if not os.path.exists(night_root):
    os.makedirs(night_root)

day_begin = dt.datetime.strptime('12:00:00', '%H:%M:%S')
day_end = dt.datetime.strptime('20:00:00', '%H:%M:%S')

night_begin = dt.datetime.strptime('00:00:00', '%H:%M:%S')
night_end = dt.datetime.strptime('06:00:00', '%H:%M:%S')


def is_day_image_pixel_based(path):
    is_day = False
    img = io.imread(path, as_grey=True)
    img_mean_px = img.mean()
    if img_mean_px > args.day_threshold:
        is_day = True
    return is_day


def get_time(path):
    """expected path format yyyymmdd_hhmmss.jpg

    Args:
        path: str

    Returns:
        rec_time: datime.time object

    """
    time_str = os.path.splitext(path)[0].split('_')[-1]
    rec_time = dt.datetime.strptime(time_str, '%H%M%S')
    return rec_time

#TODO try edge detection for eliminating constant color images
#TODO invistigate images with are imposible to open in Linux but skimage is able to read them

def create_day_night_dataset():
    for root, dir, files in os.walk(args.data_root):
        if files:
            random.shuffle(files)
            for f in files[:args.folder_num_imgs]:
                path = os.path.join(root, f)
                rec_time = get_time(path)

                try:
                    is_day = is_day_image_pixel_based(path)
                except ValueError:
                    print('Error while reading image file: {}'.format(path))
                    continue
                except Exception:
                    print('Something wrong happened while prcessing file: {}'.format(path))
                img_name_components = path.split('/')[-3:]
                img_camera = img_name_components[0]
                # img_date = '_'.join(img_name_components[1].split('.')) #not used
                img_name = img_name_components[2]

                dest_img_name = '_'.join([img_camera, img_name])
                if day_begin <= rec_time <= day_end and is_day:
                    print(rec_time, 'day')
                    dest_path = os.path.join(day_root, dest_img_name)
                    shutil.copy(path, dest_path)
                elif night_begin <= rec_time <= night_end:
                    print(rec_time, 'night')
                    dest_path = os.path.join(night_root, dest_img_name)
                    shutil.copy(path, dest_path)

if __name__ == '__main__':
    create_day_night_dataset()
