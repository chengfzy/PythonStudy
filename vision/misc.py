"""
Some Useful Function
"""

import cv2 as cv
import numpy as np
import logging


def toGray(img: np.ndarray, cloned=True):
    """
    Convert color image to gray image
    This function choose the convert code based on the type of input image,
        (1) If type is CV_8UC4, then the color order of input image is BGRA.
        (2) If type is CV_8UC3, then the color order of input image is BGR.
        (3) If type is CV_8UC1, then the color order of input image is Gray.
        (4) Other type will throw an error

    Args:
        img (np.ndarray): Input color image
        cloned (bool, optional): Whether to output an cloned image if the image type is CV_8UC1, cloned image will not
            destory the input memory. Defaults to True.

    Returns:
        np.ndarray: Gray image
    """
    if img.dtype != np.dtype('uint8'):
        logging.fatal(f'input image type {img.dtype} don\'t equal to "uint8"')

    channel = len(img.shape)
    if channel == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    elif channel == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif channel == 1:
        return img.copy() if cloned else img
    else:
        logging.fatal(f'not support image channels({channel})')


def toColor(img: np.ndarray, cloned=True):
    """
    Convert gray image to color(BGR) image
    This function choose the convert code based on the type of input image,
        (1) If type is CV_8UC4, then the color order of input image is BGRA.
        (2) If type is CV_8UC3, then the color order of input image is BGR.
        (3) If type is CV_8UC1, then the color order of input image is Gray.
        (4) Other type will throw an error

    Args:
        img (np.ndarray): Input gray image
        cloned (bool, optional): Whether to output an cloned image if the image type is CV_8UC3, cloned image will not
            destory the input memory. Defaults to True.

    Returns:
        np.ndarray: Color image
    """
    if img.dtype != np.dtype('uint8'):
        logging.fatal(f'input image type {img.dtype} don\'t equal to "uint8"')

    channel = len(img.shape)
    if channel == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    elif channel == 3:
        return img.copy() if cloned else img
    elif channel == 1:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        logging.fatal(f'not support image channels({channel})')


def imshowEx(win_name: str, img: np.ndarray, max_show_img_size=960):
    """
    Show image with max image size

    Args:
        win_name (str): Window name
        img (np.ndarray): Image
        max_show_img_size (int, optional): Max image size to show. Defaults to 960.
    """
    if max(img.shape[:2]) > max_show_img_size:
        ratio = max_show_img_size / max(img.shape[:2])
        img_show = cv.resize(img, None, fx=ratio, fy=ratio)
        cv.imshow(win_name, img_show)
    else:
        cv.imshow(win_name, img)
