import enum


class HeadingType(enum.Enum):
    """ Heading type """
    Section = 1
    SubSection = 2
    Paragraph = 3


class Heading:
    """
    Heading class, used for print some heading information
    """

    def __init__(self, text: str, break_line: bool, heading_type: HeadingType, par_len: int = 80):
        """
        Init Heading with text, break line, type and paragraph length
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        :param heading_type: Heading type
        :param par_len: Print length for Paragraph type, and the length of Section and Subsection will be 1.5 * par_len
        """
        self.__text = text
        self.__break_line = break_line
        self.__heading_type = heading_type
        self.__par_len = par_len

    def __str__(self):
        str = '\n' if self.__break_line else ''
        fill_char = '-' if self.__heading_type == HeadingType.SubSection else '='
        info_len = self.__par_len if self.__heading_type == HeadingType.Paragraph else 1.5 * self.__par_len
        if len(self.__text) == 0:
            str += info_len * fill_char
        else:
            fill_str = max(5, int((info_len - len(self.__text)) / 2)) * fill_char
            str += f'{fill_str} {self.__text} {fill_str}'

        return str


class Section(Heading):
    """ Section heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Section)


class SubSection(Heading):
    """ SubSection heading """

    def __init__(self, text: str = "", break_line: bool = True):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.SubSection)


class Paragraph(Heading):
    """ Paragraph heading """

    def __init__(self, text: str = "", break_line: bool = False):
        """
        Init Heading with text and break line
        :param text: Heading text
        :param break_line: Flag to indict whether add a new line to show this info
        """
        super().__init__(text, break_line, HeadingType.Paragraph)
