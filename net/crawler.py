import requests


def basic():
    url = "https://www.baidu.com"
    url = 'https://raw.githubusercontent.com/chengfzy/CPlusPlusStudy/master/common/include/common/Heading.hpp'
    r = requests.get(url)
    print(r.text)


if __name__ == '__main__':
    basic()