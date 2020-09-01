"""
Parse COCO dataset for selected categories and save to YOLO format

Install pycocotools before run this script
"""

from pycocotools.coco import COCO
import argparse
import shutil
import os


def convert_pos(size, box):
    """
    Convert position from COCO to YOLO

    Args:
        size (set): image size (width, height)
        box (set): ROI position in COCO format, (xMin, yMin, width, height) with pixel unit

    Returns:
        set: the position of YOLO format, (centerX, centerY, width, height), the size is in range [0, 1]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.) * dw
    y = (box[1] + box[3] / 2.) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Parse COCO dataset and save to YOLO format')
    parser.add_argument('--root', help='the root folder for COCO dataset')
    parser.add_argument('--type', type=str, default='train2017', help='the data type')
    args = parser.parse_args()
    print(args)

    # other argument
    selected_categories = ['person']
    selected_cats_id = {name: idx for idx, name in enumerate(selected_categories)}  # selected classes <name, Index>

    # some variables
    ann_file = os.path.join(args.root, f'annotations/instances_{args.type}.json')

    # create image directory if not exist
    files_dir = os.path.join(args.root, 'images')
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    # initialize COCO api for instance annotations
    coco = COCO(ann_file)
    # get all images containing given categories
    cat_ids = coco.getCatIds(catNms=selected_categories)
    img_ids = coco.getImgIds(catIds=cat_ids)
    images_file = open(os.path.join(files_dir, f'{args.type}.txt'), 'w')  # file to save image path

    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        filename = img['file_name']
        width = img['width']
        height = img['height']
        print(f'Image ID: {img_id}, filename: {filename}, width: {width}, height: {height}')
        # print(f'Image: {img}')
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        # print(f'annotation IDs: {ann_ids}')

        # write image names
        images_file.write(f"{os.path.join(args.root, args.type, filename)}\n")

        # write labels
        labels_file = open(os.path.join(args.root, args.type, f'{filename[:-4]}.txt'), 'w')
        for ann_id in ann_ids:
            anns = coco.loadAnns(ann_id)[0]
            cat_id = anns['category_id']
            cat = coco.loadCats(cat_id)[0]['name']
            class_id = selected_cats_id[cat]
            box = anns['bbox']
            pos = convert_pos((width, height), box)
            labels_file.write(f"{class_id} {' '.join(str(p) for p in pos)}\n")
        labels_file.close()

    images_file.close()


if __name__ == "__main__":
    main()