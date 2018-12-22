from unittest import TestCase

import qiqc.preprocessors as QP


class TestSpacingPunctReplacer(TestCase):

    def test_call(self):
        replacer = QP.SpacingPunctReplacer()
        old = 'This is, a pen.(aa)'
        new = 'This is ,  a pen .  ( aa ) '
        self.assertEqual(new, replacer(old))


class TestNumberReplacer(TestCase):

    def test_call(self):
        replacer = QP.NumberReplacer()
        old = 'in 1990s(90s)'
        new = 'in ####s(##s)'
        self.assertEqual(new, replacer(old))


class TestHengZhengMispellReplacer(TestCase):

    def test_call(self):
        replacer = QP.HengZhengMispellReplacer()
        old = "I can't play tennis."
        new = "I cannot play tennis."
        self.assertEqual(new, replacer(old))
