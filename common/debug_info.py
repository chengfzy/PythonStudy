"""
Debug Info
"""


def section(title, break_line=True):
    """
    Obtain section string
    :param title: title string
    :param break_line: break new line to write the section
    :return: section string
    """
    space = '=' * int((120 - len(title)) / 2)
    line = '\n' if break_line else ''
    line += space + " " + title + " " + space
    return line


def sub_section(title, break_line=False):
    """
    Obtain sub section string
    :param title: title string
    :param break_line: break new line to write the section
    :return: sub-section string
    """
    space = '-' * int((120 - len(title)) / 2)
    line = '\n' if break_line else ''
    line += space + " " + title + " " + space
    return line
