from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join
from tqdm import tqdm

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break


def export_images(db_path, out_dir, n_images, flat=False):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in tqdm(cursor, total=n_images):
            if not flat:
                image_out_dir = join(out_dir, '/'.join(key[:6].decode()))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key.decode() + '.png')
            img = cv2.imdecode(numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imwrite(image_out_path, img)
            count += 1
            if count == n_images:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    parser.add_argument('--n_images', type=int, default=210526) # Gives exactly 200K training after a 95/5 split
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path

    for lmdb_path in lmdb_paths:
        if command == 'view':
            view(lmdb_path)
        elif command == 'export':
            export_images(lmdb_path, args.out_dir, args.n_images, args.flat)


if __name__ == '__main__':
    main()
