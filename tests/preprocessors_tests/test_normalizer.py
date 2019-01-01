from unittest import TestCase

import qiqc.preprocessors as QP


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


class TestMisspellReplacer(TestCase):

    def test_call(self):
        replacer = QP.MisspellReplacer()
        for old, new in replacer.rule.items():
            self.assertEqual(new, replacer(old))
