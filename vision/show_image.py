"""
Show image in folder
"""
import argparse
import imghdr
import os

import cv2 as cv


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Show Image in Folder')
    parser.add_argument('--folder', required=True, help='image folder')
    args = parser.parse_args()
    print(args)

    # list all images
    files = [f for f in os.listdir(args.folder) if imghdr.what(os.path.join(args.folder, f)) is not None]
    files.sort()

    # read image and show
    wait_time = 100
    i = 0
    while i < len(files):
        # read image and resize if it's too large
        image = cv.imread(os.path.join(args.folder, files[i]), cv.IMREAD_UNCHANGED)
        if max(image.shape) > 500:
            ratio = 500 / max(image.shape)
            image = cv.resize(image, (-1, -1), fx=ratio, fy=ratio)

        # draw index and file name
        cv.putText(image, f'[{i}/{len(files)}] {files[i]}', (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
                   cv.LINE_AA)

        # show image
        cv.imshow('Image', image)
        key = cv.waitKey(wait_time)
        if key == ord('w') or key == ord('W'):  # faster
            wait_time = max(10, wait_time - 50)
        elif key == ord('s') or key == ord('S'):  # slower
            wait_time += 50
        elif key == ord('a') or key == ord('A'):  # last
            i = max(0, i - 5)
        elif key == ord('a') or key == ord('A'):  # next
            i += 5
        elif key == 27 or key == ord('x') or key == ord('X'):  # exit
            break
        elif key == ord(' '):  # pause
            if wait_time > 0:
                last_wait_time = wait_time
                wait_time = 0
            else:
                wait_time = last_wait_time
        else:
            i += 1


if __name__ == '__main__':
    main()
