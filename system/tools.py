"""
Some System Function
"""
import os
import shutil
from colorama import Fore


def get_folder_size(folder: str):
    """
    Get the folder size (MB)

    Args:
        folder (str): Input folder

    Returns:
        float: Folder size, MB
    """
    size = 0
    for root, _, files in os.walk(folder):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return size / 1024**2  # MB


def safe_rm(folder: str):
    """
    Safely remove some folder

    Args:
        folder (str): Input folder

    Raises:
        SystemError: If the folder size is large than some threshold, it will raise this error
    """
    if get_folder_size(folder) > 1000:
        print(f'{Fore.RED}The folder "{folder}" has many document with size = {get_folder_size(folder):.3f} MB')
        ret = input(f'Are you still delete it? (yes/no){Fore.RESET} ')
        if ret != 'yes':
            raise SystemError('exit')

    shutil.rmtree(folder, ignore_errors=True)
