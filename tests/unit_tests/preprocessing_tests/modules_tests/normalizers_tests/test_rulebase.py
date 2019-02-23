from unittest import TestCase

import qiqc.preprocessing.modules as QP


class TestPunctSpacer(TestCase):

    def test_call(self):
        replacer = QP.PunctSpacer()
        old = 'This is, a pen.(aa)'
        new = 'This is ,  a pen .  ( aa ) '
        self.assertEqual(new, replacer(old))


class TestNumberReplacer(TestCase):

    def test_call(self):
        replacer = QP.NumberReplacer()
        old = 'in 1990s(90s)'
        new = 'in ####s(##s)'
        self.assertEqual(new, replacer(old))

        replacer = QP.NumberReplacer(with_underscore=True)
        old = 'in 1990s(90s)'
        new = 'in  __####__ s( __##__ s)'
        self.assertEqual(new, replacer(old))


class TestMisspellReplacer(TestCase):

    def test_call(self):
        replacer = QP.MisspellReplacer()
        for old, new in replacer.rule.items():
            self.assertEqual(new, replacer(old))


class TestUnidecodeWeak(TestCase):

    def test_call(self):
        replacer = QP.unidecode_weak
        old = 'what\u200b is √3?'
        new = 'what  is  √ 3?'
        self.assertEqual(new, replacer(old))
