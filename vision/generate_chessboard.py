"""
Generate chessboard and save into pdf
"""

import pyx
import argparse


def generate_chessboard(c, nx, ny, szx, szy):
    # convert to cm
    szx = szx * 100
    szy = szy * 100

    print(f'Generate chessboard with {nx}x{ny} and a box size of {szx}x{szy} cm')

    # draw boxes
    for x in range(nx + 1):
        for y in range(ny + 1):
            # origin point (top left)
            x0 = x * szx
            y0 = y * szy
            if (x + y + 1) % 2 != 0:
                c.fill(pyx.path.rect(x0, y0, szx, szy), [pyx.color.rgb.black])

    # add caption
    c.text(1.05 * szx, 0.04 * szy, f'{nx}x{ny}@{szx}x{szy}cm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate chessboard and save into .PDF file')
    parser.add_argument('--file', type=str, default='../data/chessboard', dest='file', help='output file name')
    parser.add_argument('--nx', type=int, default=6, dest='nx',
                        help='the number of tags in x direction (default: %(default)s)')
    parser.add_argument('--ny', type=int, default=6, dest='ny',
                        help='the number of tags in y direction (default: %(default)s)')
    parser.add_argument('--szx', type=float, default=0.05, dest='szx',
                        help='the size of one chessboard square in x direction [m] (default: %(default)s)')
    parser.add_argument('--szy', type=float, default=0.05, dest='szy',
                        help='the size of one chessboard square in y direction [m] (default: %(default)s)')
    args = parser.parse_args()

    # create canvas
    c = pyx.canvas.canvas()
    # generate chessboard
    generate_chessboard(c, args.nx, args.ny, args.szx, args.szy)
    # write to file
    c.writePDFfile(args.file)
