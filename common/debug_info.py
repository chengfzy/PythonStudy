"""
Debug Info
"""


def section(title: str, break_line=True):
    """
    Obtain section string
    :param title: title string
    :param break_line: break new line to write the section
    :return: section string
    """
    line = '\n' if break_line else ''
    if len(title) == 0:
        line += '=' * 120
    else:
        space = '=' * int((120 - len(title)) / 2)
        line += space + " " + title + " " + space
    return line


def sub_section(title: str, break_line=True):
    """
    Obtain sub section string
    :param title: title string
    :param break_line: break new line to write the section
    :return: sub-section string
    """
    line = '\n' if break_line else ''
    if len(title) == 0:
        line += '-' * 120
    else:
        space = '-' * int((120 - len(title)) / 2)
        line += space + " " + title + " " + space
    return line


def paragraph(title: str):
    """
    Obtain paragraph string
    :param title: title string
    :return: sub-section string
    """
    if len(title) == 0:
        return '=' * 60
    else:
        space = '=' * int((60 - len(title)) / 2)
        line = space + " " + title + " " + space
        return line
