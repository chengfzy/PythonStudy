"""
Show images in folder
"""
import argparse
import imghdr
from pathlib import Path
import cv2 as cv


def show(folder: Path, max_size: int = 1080, wait_time: int = 30):
    """
    Show image

    Args:
        folder (Path): Image folder path
        max_size (int): max image size
        wait_time (int): wait time, unit: ms
    """
    # list all images
    files = [f for f in folder.iterdir() if f.is_file() and imghdr.what(f) is not None]
    files.sort(key=lambda v: int(v.stem.split('_')[0]))

    # read image and show
    wait_time = wait_time
    step = 1
    i = 0
    while i < len(files):
        # read image and resize if it's too large
        image = cv.imread(str(files[i]), cv.IMREAD_UNCHANGED)
        if max(image.shape) > max_size:
            ratio = max_size / max(image.shape)
            image = cv.resize(image, (-1, -1), fx=ratio, fy=ratio)

        # draw index and file name
        cv.putText(image, f'[{i}/{len(files)}] {files[i].name}', (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
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
        elif key == ord('a') or key == ord('A'):  # move backward fast
            i = max(0, i - 5 * step)
        elif key == ord('d') or key == ord('D'):  # move forward fast
            i += 1
        elif key == ord('h') or key == ord('H'):  # last image
            i = max(0, i - 1)
        elif key == ord('j') or key == ord('J'):  # next image
            i += 1
        elif key == 27 or key == ord('x') or key == ord('X') or key == ord('q') or key == ord('Q'):  # exit
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
    # argument parser
    help_str = \
    '''
Show Images in Folder. Key command:
    W/w:        Show faster
    S/s:        Show slower
    A/a:        Move backward fast
    D/d:        Move forward fast
    H/h:        Last image
    J/j:        Next image
    Q/q/X/x:    Quit
    Space:      Pause
    '''
    parser = argparse.ArgumentParser(description=help_str, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('folder', help='image folder')
    parser.add_argument('--max-size', type=int, default=1080, help='max image size to show')
    parser.add_argument('--wait-time', type=int, default=30, help='initial wait time(ms)')
    args = parser.parse_args()
    print(args)

    show(Path(args.folder).absolute(), args.max_size, args.wait_time)
