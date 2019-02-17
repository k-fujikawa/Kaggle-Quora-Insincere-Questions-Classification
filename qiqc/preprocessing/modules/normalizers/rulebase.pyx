# cython: language_level=3
import re

import numpy as np
cimport numpy as np


cdef class StringReplacer:
    cpdef public dict rule
    cpdef list keys
    cpdef list values
    cpdef int n_rules

    def __init__(self, dict rule):
        self.rule = rule
        self.keys = list(rule.keys())
        self.values = list(rule.values())
        self.n_rules = len(rule)

    def __call__(self, str x):
        cdef int i
        for i in range(self.n_rules):
            if self.keys[i] in x:
                x = x.replace(self.keys[i], self.values[i])
        return x

    def __getstate__(self):
        return (self.rule, self.keys, self.values, self.n_rules)

    def __setstate__(self, state):
        self.rule, self.keys, self.values, self.n_rules = state


cdef class RegExpReplacer:
    cdef dict rule
    cdef list keys
    cdef list values
    cdef regexp
    cdef int n_rules

    def __init__(self, dict rule):
        self.rule = rule
        self.keys = list(rule.keys())
        self.values = list(rule.values())
        self.regexp = re.compile('(%s)' % '|'.join(self.keys))
        self.n_rules = len(rule)

    @property
    def rule(self):
        return self.rule

    def __call__(self, str x):
        def replace(match):
            x = match.group(0)
            if x in self.rule:
                return self.rule[x]
            else:
                for i in range(self.n_rules):
                    x = re.sub(self.keys[i], self.values[i], x)
                return x
        return self.regexp.sub(replace, x)


cpdef str cylower(str x):
    return x.lower()


Cache = {}
is_alphabet = re.compile(r'[a-zA-Z]')


cpdef str unidecode_weak(str string):
    """Transliterate an Unicode object into an ASCII string
    >>> unidecode(u"\u5317\u4EB0")
    "Bei Jing "
    """

    cdef list retval = []
    cdef int i = 0
    cdef int n = len(string)
    cdef str char

    for i in range(n):
        char = string[i]
        codepoint = ord(char)

        if codepoint < 0x80: # Basic ASCII
            retval.append(char)
            continue

        if codepoint > 0xeffff:
            continue  # Characters in Private Use Area and above are ignored

        section = codepoint >> 8   # Chop off the last two hex digits
        position = codepoint % 256 # Last two hex digits

        try:
            table = Cache[section]
        except KeyError:
            try:
                mod = __import__('unidecode.x%03x'%(section), [], [], ['data'])
            except ImportError:
                Cache[section] = None
                continue   # No match: ignore this character and carry on.

            Cache[section] = table = mod.data

        if table and len(table) > position:
            if table[position] == '[?]' or is_alphabet.match(table[position]):
                retval.append(' ' + char + ' ')
            else:
                retval.append(table[position])

    return ''.join(retval)
