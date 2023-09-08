"""
Rename files in folder
"""

import argparse
import os
import shutil


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Rename file names in folder')
    parser.add_argument('--folder', required=True, help='input folder')
    args = parser.parse_args()
    print(args)

    # here I give an example to rename the file name
    for file in os.listdir(args.folder):
        if file.endswith('.jpg'):
            time = int(file.replace('.jpg', ''))
            time -= 28800 * 1000000000
            new_file = f'{time}.jpg'
            print(f'{file} => {new_file}')
            shutil.move(os.path.join(args.folder, file), os.path.join(args.folder, new_file))


if __name__ == "__main__":
    main()
