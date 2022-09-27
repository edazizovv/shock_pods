#


#
import pandas


#


#
def is_number(symbol):
    numbers = '1234567890'
    return symbol in numbers


def is_symbol(symbol):
    symbols = ',.;:-?!/\\()"' + "'"
    return symbol in symbols


def is_latin(symbol):
    latin = 'abcdefghijklmnopqrstuvwxyz'
    return symbol.lower() in latin


def is_cyrillic(symbol):
    cyrillic = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    return symbol.lower() in cyrillic


def is_spaces(symbol):
    spaces = ' \n'
    return symbol in spaces


def is_fine(symbol):
    return any([is_number(symbol),
                is_symbol(symbol),
                is_latin(symbol),
                is_cyrillic(symbol),
                is_spaces(symbol)])


def treat_text(text):
    result = ''
    for x in text:
        if is_fine(x):
            result += x
    result = ' '.join(result.replace('\n', ' ').split())

    return result


def joint_text(row):
    if pandas.isna(row['text']) and pandas.isna(row['caption']):
        return ''
    elif (not pandas.isna(row['text'])) and (not pandas.isna(row['caption'])):
        return treat_text(row['text'] + row['caption'])
    elif pandas.isna(row['text']) and (not pandas.isna(row['caption'])):
        return treat_text(row['caption'])
    else:
        return treat_text(row['text'])


def no_cyrillic(row):
    return not any([is_cyrillic(x) for x in row])


def headlines_specific(x):
    if 'source:' in x:
        return x[:x.index('source:')]
    else:
        return x


data = pandas.read_csv('./headlines_en.csv')
treated = data.apply(func=joint_text, axis=1)
treated = treated.apply(func=headlines_specific)
treated = treated[(treated != '') & treated.apply(func=no_cyrillic)].copy()
treated.name = 'text'
treated.to_csv('./headlines_en_clean.csv', index=False)
