'''
Borrowed
from https://github.com/keithito/tacotron/blob/master/text/numbers.py
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/g2p  (MIT License)

Modified, you can do a git diff to see that. 

I adapted to take into account a list of currencies instead of just dollars and pounds
'''
from __future__ import print_function
import inflect
import re

import pandas as pd

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')

_percent_re = re.compile(r'\%([0-9\,]*[0-9]+)')
_percent_re2 = re.compile(r'([0-9\,]*[0-9]+)\%')

# _pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_dollars_re2 = re.compile(r'([0-9\.\,]*[0-9]+)\$')
# _euros_re = re.compile(r'\€([0-9\.\,]*[0-9]+)')
# _euros_re2 = re.compile(r'([0-9\.\,]*[0-9]+)\€')

_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

# File coming from https://gist.github.com/manishtiwari25/d3984385b1cb200b98bcde6902671599
currencies_df = pd.read_json('data/world_currency_symbols.json')

symbols_name_df = currencies_df[['Symbol', 'Currency']].drop_duplicates()
symbols_name_df.Currency = symbols_name_df.Currency.str.lower()

names = symbols_name_df.Currency.unique()
symbols = symbols_name_df.Symbol.unique()

names_symbols_dict = {
    name: symbols_name_df[symbols_name_df.Currency == name].Symbol.tolist()
    for name in names
}
symbols_names_dict = {
    symbol: symbols_name_df[symbols_name_df.Symbol == symbol].Currency.tolist()
    for symbol in symbols
}

symbols_names_dict["$"] = ['dollar', 'peso']
symbols_names_dict["¥"] = ['yen', 'yuan']

currency_regexes = {}
base_str1 = r'€([0-9\.\,]*[0-9]+)'
base_str2 = r'([0-9\.\,]*[0-9]+)€'
for s in symbols_names_dict:
    # I only take the one with 1 character so that it is actually a symbol. If I want to treat others, I would have to adapt the regexes probably
    if len(s) == 1:
        if (
            s != "$"
        ):  # $ is a symbol that is used in regex, so we do it separately, adding a blackslash to escape it
            currency_regexes[s] = []
            currency_regexes[s].append(base_str1.replace('€', s))
            currency_regexes[s].append(base_str2.replace('€', s))


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_currency(match, currency_name='dollar'):
    # match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' ' + currency_name + 's'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = currency_name if dollars == 1 else currency_name + 's'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s and %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = currency_name if dollars == 1 else currency_name + 's'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero ' + currency_name + 's'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(
                ', ', ' '
            )
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)

    text = re.sub(_percent_re, r'\1 percent', text)
    text = re.sub(_percent_re2, r'\1 percent', text)

    # dollars done alone, because the symbol is part of regex syntax, so an additional backslash to escape is there
    text = re.sub(_dollars_re, lambda x: _expand_currency(x.group(1), 'dollar'), text)
    text = re.sub(_dollars_re2, lambda x: _expand_currency(x.group(1), 'dollar'), text)

    for s in currency_regexes:
        for el in currency_regexes[s]:
            text = re.sub(
                el, lambda x: _expand_currency(x.group(1), symbols_names_dict[s][0]), text
            )

    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
