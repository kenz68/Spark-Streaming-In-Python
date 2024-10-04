import unittest

from datasketch import HyperLogLog
from test.utils import fake_hash_func


class TestHyperLogLog(unittest.TestCase):
    _class = HyperLogLog
    def test_init(self):
        h = self._class(4, hashfunc=fake_hash_func)
        self.assertEqual(h.m, 1 << 4)
        self.assertEqual(len(h.reg), h.m)
        self.assertTrue(all(0 == i for i in h.reg))

if __name__ == '__main__':
    unittest.main()
