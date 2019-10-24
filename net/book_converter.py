"""
Book Converter

Convert a book in website with html format to PDF
"""

import requests
import pdfkit
import re
import os


class Converter:
    """
    Convert book in html to pdf
    """

    def __init__(self, main_page: str, save_folder: str):
        self.main_page = main_page
        self.headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6,ja;q=0.5',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        self.folder = save_folder

    def run(self):
        """
        The main run function
        """
        # make save folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        # all htmls to convert to pdf
        htmls = []

        # parse and save main page
        page_file = os.path.join(self.folder, 'main.html')
        print(f'add main page: {page_file}')
        # save main page and images to file
        if not os.path.exists(page_file):
            response = requests.get(self.main_page, headers=self.headers)
            content = response.content.decode('utf-8')
            self.save_page_img(self.main_page, content, self.folder)
            self.save_html(content, page_file)
        htmls.append(page_file)

        # get section urls
        sections = self.parse_main(self.main_page)
        # remove duplicate item
        sections = list(dict.fromkeys(sections))

        # parse and save section urls
        for idx, url in enumerate(sections):
            print(f'[{idx}/{len(sections)}] {url}')
            # make section folder
            p = url.replace(self.main_page, "")
            folder = os.path.join(self.folder, p[: p.rfind('/')])
            os.makedirs(folder, exist_ok=True)
            htmls.append(self.process_section(url, folder))

        # save to pdf
        print('\nConvert pages to PDF')
        self.convert_pdf(htmls, "../spline.pdf")

    def parse_main(self, url):
        """
        Parse main page, obtain section urls
        """
        response = requests.get(url, headers=self.headers)
        content = response.content.decode('utf-8')
        pattern = "<A HREF=\"(.*?)\""
        domain = url[:url.rfind('/') + 1]
        sections = []
        for m in re.compile(pattern).findall(content):
            if m.endswith("html"):
                sections.append(domain + m)
        return sections

    def process_section(self, url: str, folder: str):
        """
        Process section page
        :param url: Section url
        :param folder: Save folder for the section page, should be the actual subfolder which will contain the html file
        :return: Saved html file (full) path
        """
        page_file = os.path.join(folder, url[url.rfind('/') + 1:])
        print(f'\tadd section page: {page_file}')
        if not os.path.exists(page_file):
            response = requests.get(url, headers=self.headers)
            content = response.content.decode('utf-8')
            self.save_page_img(url, content, folder)
            self.save_html(content, page_file)

        return page_file

    def save_page_img(self, url: str, content: str, folder: str):
        """
        Save page images to folder
        """
        pattern = "<IMG SRC=\"(.*?)\""
        domain = url[:url.rfind('/') + 1]
        image_urls = re.compile(pattern).findall(content)
        for m in image_urls:
            if not m.startswith("http"):
                img_file = os.path.join(folder, m)
                if not os.path.exists(img_file):
                    img = requests.get(domain + m)
                    with open(img_file, 'ab') as f:
                        f.write(img.content)

    def save_html(self, content: str, file_name="page.html"):
        """Save page content to html"""
        with open(file_name, 'w') as f:
            f.write(content)

    def convert_pdf(self, htmls: list, file_name='out.pdf'):
        """Convert all htmls to pdf"""
        options = {
            'page-size': 'A4',
            'margin-top': '30mm',
            'margin-right': '30mm',
            'margin-bottom': '30mm',
            'margin-left': '30mm',
            'header-center': '[section]',
            'footer-center': '[page]/[topage]',
            'encoding': "UTF-8",
            'custom-header': [
                ('Accept-Encoding', 'gzip')
            ],
            'cookie': [
                ('cookie-name1', 'cookie-value1'),
                ('cookie-name2', 'cookie-value2'),
            ]
        }
        pdfkit.from_file(htmls, os.path.join(self.folder, file_name), options=options)


if __name__ == '__main__':
    converter = Converter('https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/', '../data/spline')
    converter.run()
