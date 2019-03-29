import unittest
import utils as ren

index = [
    {
        'in_pth': 'source/how-residual-shortcuts-speed-up-learning/how-residual-shortcuts-speed-up-learning.ipynb',
        'type': 'article',
        'format': 'ipynb',
        'publish': True
    },
    {
        'in_pth': 'https://github.com/a-martyn/pix2pix/',
        'type': 'project',
        'format': 'url',
        'publish': False
    },
    {
        'in_pth': 'source/ml-curriculum/ml-curriculum.md',
        'type': 'article',
        'format': 'md',
        'publish': True
    },
    {
        'in_pth': 'source/ml-learning-log/ml-learning-log.md',
        'type': 'article',
        'format': 'md',
        'publish': True
    },
]

class Tests(unittest.TestCase):

    def test_filter(self):
        self.assertEqual(len(ren.filter('publish', False, index)), 1)
        self.assertEqual(len(ren.filter('format', 'md', index)), 2)

    def test_add_out_pth(self):
        result = ren.add_out_pth('./output', index)
        self.assertEqual(result[1]['out_pth'], 
                         'https://github.com/a-martyn/pix2pix/')

    def test_assets_validate(self):
        assets = ren.Assets(index, ['md', 'ipynb'], '')
        # Happy path
        assets.assets = [('a', 'x/y/a'), ('b', 'z/x/b')]
        assets.validate()
        # Failure mode: duplicated filename
        assets.assets = [('a', 'x/y/a'), ('a', 'z/x/b')]
        with self.assertRaises(NameError):
            assets.validate()

suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)