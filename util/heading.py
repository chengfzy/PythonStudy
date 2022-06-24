import enum
from colorama import Fore, Back, Style


class HeadingType(enum.Enum):
    """ Heading type """
    Title = 1
    Section = 2
    SubSection = 3
    Paragraph = 4


class Heading:
    """
    Heading class, used for print some heading information
    """

    def __init__(self, text: str, break_line: bool, heading_type: HeadingType, sec_len: int = 100):
        """
        Init Heading with text, break line, type and paragraph length
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        :param heading_type: Heading type
        :param sec_len: Print length for section and subsection type, and the length of paragraph will be SecLen / 1.5
        """
        self._text = text
        self._break_line = break_line
        self._heading_type = heading_type
        self._sec_len = sec_len


class Title(Heading):
    """ Title heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Title)

    def __str__(self):
        info = '\n' if self._break_line else ''

        info_len = max(self._sec_len - 2, len(self._text) + 10)
        info += f'{Fore.CYAN}{Style.BRIGHT}╔{"":═^{info_len}}╗\n'
        info += f'║{"":^{info_len}}║\n'
        info += f'║{" " + self._text + " ":^{info_len}}║\n'
        info += f'║{"":{info_len}}║\n'
        info += f'╚{"":═^{info_len}}╝{Style.RESET_ALL}\n'
        return info


class Section(Heading):
    """ Section heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Section)

    def __str__(self):
        info = '\n' if self._break_line else ''
        if len(self._text) == 0:
            info += f'{Fore.CYAN}{"":═^{self._sec_len}}'
        else:
            info += f'{Fore.CYAN}{" " + self._text + " ":═^{max(self._sec_len, len(self._text) + 12)}}'
        info += f'{Style.RESET_ALL}'
        return info


class SubSection(Heading):
    """ SubSection heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.SubSection)

    def __str__(self):
        info = '\n' if self._break_line else ''
        if len(self._text) == 0:
            info += f'{Fore.CYAN}{"":━^{self._sec_len}}'
        else:
            info += f'{Fore.CYAN}{" " + self._text + " ":━^{max(self._sec_len, len(self._text) + 12)}}'
        info += f'{Style.RESET_ALL}'
        return info


class Paragraph(Heading):
    """ Paragraph heading """

    def __init__(self, text: str = "", break_line: bool = False):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Paragraph)

    def __str__(self):
        info = '\n' if self._break_line else ''
        if len(self._text) == 0:
            info += f'{Fore.CYAN}{"":─^{int(self._sec_len / 1.5)}}'
        else:
            info += f'{Fore.CYAN}{" " + self._text + " ":─^{max(int(self._sec_len / 1.5), len(self._text) + 12)}}'
        info += f'{Style.RESET_ALL}'
        return info
