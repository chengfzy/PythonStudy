from PyPDF2 import PdfFileReader


file = 'xxx.pdf'

chars = [f'{v}' for v in range(0, 10)]
chars.extend(['X', 'x'])
print(f'chars: {chars}')

reader = PdfFileReader(file)
for c1 in chars:
    for c2 in chars:
        for c3 in chars:
            for c4 in chars:
                password = f'{c1}{c2}{c3}{c4}'
                print(f'try {password}')
                if reader.decrypt(password):
                    print(f'success, password {password}')
                    exit()
