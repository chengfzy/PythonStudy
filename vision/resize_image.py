"""
Read image in folder, resize it and save to file with .jpg format
"""

import os
import argparse
import imghdr
import cv2 as cv


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Resize Image')
    parser.add_argument('--folder', type=str, required=True, help='image folder')
    parser.add_argument('--max_size', default=720, help='max image size to show')
    parser.add_argument('--save_folder', type=str, help='save folder, if empty it will be "folder"')
    args = parser.parse_args()
    print(args)

    # list all images
    files = [f for f in os.listdir(args.folder) if imghdr.what(os.path.join(args.folder, f)) is not None]
    files.sort()

    # save folder
    save_folder = args.folder if args.save_folder is None else args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save parameters
    save_params = (cv.IMWRITE_JPEG_QUALITY, 95)

    # read image and show
    for f in files:
        # read
        img = cv.imread(os.path.join(args.folder, f), cv.IMREAD_UNCHANGED)

        # resize
        if max(img.shape) > args.max_size:
            ratio = args.max_size / max(img.shape)
            img = cv.resize(img, (-1, -1), fx=ratio, fy=ratio)

        # save to file
        cv.imwrite(os.path.join(save_folder, f[:-4] + '.jpg'), img, save_params)


if __name__ == '__main__':
    main()
