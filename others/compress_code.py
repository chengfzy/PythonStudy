"""
Backup Code
"""

from pathlib import Path
import argparse, logging, coloredlogs, datetime
import os, subprocess, shutil
from typing import List


class Compressor:

    def __init__(self, code_folder: Path = None, save_folder: Path = None) -> None:
        self.code_folder: Path = code_folder.absolute()
        self.save_folder: Path = save_folder.absolute()

        self.log_path: Path = self.save_folder / 'CompressCode.log'
        # folder to back up build/data folders which don't need to compress
        self.bak_folder: Path = self.code_folder / 'bak'

        self.n = 0

    def run(self) -> None:
        logging.info(f'code folder: {self.code_folder}')

        # create save folder
        logging.info(f'root save folder: {self.save_folder}\n')
        # create log file
        logging.info(f'create log file: {self.log_path}')
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        self.bak_folder.mkdir(exist_ok=True, parents=True)
        self.log_file = open(self.log_path, 'w')

        for p in sorted(self.code_folder.iterdir()):
            if p.is_file():
                self.__backup_file(p)
            elif p.is_dir() and p != self.bak_folder:
                self.__process(p)

        # remove bak folder
        self.bak_folder.rmdir()

    def __process(self, folder: Path) -> bool:
        if self.__has_code(folder):
            for p in sorted(folder.iterdir()):
                if p.is_file():
                    self.__backup_file(p)
                elif p.is_dir():
                    if self.__is_code(p):
                        self.__compress_folder(p)
                    else:
                        self.__process(p)
        else:
            self.__compress_folder(folder)

    def __has_code(self, folder: Path) -> bool:
        """Check the folder has code, i.e., contains at least on .git folder"""
        if self.__is_code(folder):
            return True
        for v in folder.iterdir():
            if v.is_dir() and (self.__is_code(folder) or self.__has_code(v)):
                return True
        return False

    def __is_code(self, folder: Path) -> bool:
        """Check current folder is code folder, i.e., './.git' folder exists"""
        return (folder / '.git').exists()

    def __compress_folder(self, folder: Path) -> None:
        # by pass some temp folder
        if folder.name == '.vscode' or folder.name == '.history':
            logging.info(f'bypass {folder}...\n')
            return

        logging.info(f'[{self.n:02d}] compress {folder}...')
        self.n += 1

        # create save folder
        save_folder = self.__create_save_folder(folder)

        # don't compress build/data folder
        for p in folder.iterdir():
            # if p.is_dir() and (str(p.name).startswith('build') or (str(p.name).startswith('data'))):
            if p.is_dir() and str(p.name).startswith('build'):
                new_folder = self.bak_folder / p.name
                os.rename(p, new_folder)
                logging.info(f'backup: {p} => {new_folder}')

        # save file
        save_file = save_folder / f'{folder.name}.7z'  # or .zip
        logging.info(f'{folder} => {save_file}')

        # git pull and gc
        if self.__is_code(folder):
            process = subprocess.run(f'cd {folder}', shell=True, stdout=self.log_file, stderr=self.log_file)
            process.returncode
            # git pull
            process = subprocess.run('git pull --all', shell=True, stdout=self.log_file, stderr=self.log_file)
            if process.returncode != 0:
                logging.error(f'git pull fail, {process.returncode}')
            else:
                logging.info(f'git pull')
            # gc
            process = subprocess.run('git gc --prune=now --aggressive',
                                     shell=True,
                                     stdout=self.log_file,
                                     stderr=self.log_file)
            if process.returncode != 0:
                logging.error(f'git gc fail, {process.returncode}')
            else:
                logging.info(f'git gc')

        # compress
        cmd = f'cd {folder} && 7z a -t7z -r {save_file} {folder}'
        # cmd = f'cd {folder.parent} && zip -Dr save_file ./{folder}'
        logging.info(f'compress command: {cmd}')
        process = subprocess.run(cmd, shell=True, stdout=self.log_file, stderr=self.log_file)
        # restore backup folder
        for p in self.bak_folder.iterdir():
            if p.is_dir():
                bak_folder = folder / p.name
                os.rename(p, bak_folder)
                logging.info(f'restore: {p} => {bak_folder}')

        # print status
        if process.returncode != 0:
            logging.error(f'Error, {process.returncode}')
        else:
            logging.info('OK\n')

    def __backup_file(self, file: Path) -> None:
        # create save folder
        save_folder = self.__create_save_folder(file)

        # back file
        new_file = save_folder / file.name
        shutil.copy(file, new_file, follow_symlinks=False)
        logging.info(f'[{self.n:02d}] backup file: {file} => {new_file}\n')
        self.n += 1

    def __create_save_folder(self, p: Path) -> Path:
        save_folder = self.save_folder / str(p.parent).removeprefix(str(self.code_folder)).removeprefix('/')
        if not save_folder.exists():
            logging.info(f'create save folder: {save_folder}')
            save_folder.mkdir(exist_ok=True, parents=True)
        return save_folder


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Backup Code')
    parser.add_argument('code_folder', help='code folder')
    parser.add_argument('--save-folder', default='./data/CompressedCode', help='folder to save compressed code files')
    args = parser.parse_args()
    print(args)

    # config logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"/tmp/{Path(__file__).stem}.{datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.log")
        ])
    coloredlogs.install(fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    compressor = Compressor(Path(args.code_folder), Path(args.save_folder))
    compressor.run()