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
    parser.add_argument('--max_size', default=1000, help='max image size to show')
    parser.add_argument('--wait_time', type=int, default=100, help='initial wait time(ms)')
    args = parser.parse_args()
    print(args)

    # list all images
    files = [f for f in os.listdir(args.folder) if imghdr.what(os.path.join(args.folder, f)) is not None]
    files.sort()

    # read image and show
    wait_time = args.wait_time
    step = 1
    i = 0
    while i < len(files):
        # read image and resize if it's too large
        image = cv.imread(os.path.join(args.folder, files[i]), cv.IMREAD_UNCHANGED)
        if max(image.shape) > args.max_size:
            ratio = args.max_size / max(image.shape)
            image = cv.resize(image, (-1, -1), fx=ratio, fy=ratio)

        # draw index and file name
        cv.putText(image, f'[{i}/{len(files)}] {files[i]}', (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
                   cv.LINE_AA)

        # show image
        cv.imshow('Image', image)
        key = cv.waitKey(wait_time)
        if key == ord('w') or key == ord('W'):  # faster
            if wait_time > 5:
                wait_time = max(5, wait_time - 10)
            else:
                step += 1
        elif key == ord('s') or key == ord('S'):  # slower
            if step > 1:
                step -= 1
            else:
                wait_time += 10
        elif key == ord('a') or key == ord('A'):  # last
            i = max(0, i - 5 * step)
        elif key == ord('a') or key == ord('A'):  # next
            i += 5 * step
        elif key == 27 or key == ord('x') or key == ord('X'):  # exit
            break
        elif key == ord(' '):  # pause
            if wait_time > 0:
                last_wait_time = wait_time
                wait_time = 0
            else:
                wait_time = last_wait_time
        else:
            i += step


if __name__ == '__main__':
    main()
