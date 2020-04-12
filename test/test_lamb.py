import re
from unittest import TestCase

from t2b.lamb import make_packet, lambda_handler
from .test_t2b import GeneralTest


class Test(TestCase, GeneralTest):

    def test_normal(self):
        for filename in self.images:
            test_id = int(re.match(".+test(\d).+", filename).groups()[0])
            packet = make_packet(filename, test_id)
            res = lambda_handler(packet)
            pass
