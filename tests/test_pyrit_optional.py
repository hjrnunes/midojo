"""The optional route never makes PyRIT a core import dependency."""

import subprocess
import sys


def test_core_imports_without_pyrit_and_optional_import_names_the_extra():
    code = """
import importlib.abc
import sys

class NoPyrit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyrit' or fullname.startswith('pyrit.'):
            raise ModuleNotFoundError('PyRIT intentionally unavailable', name='pyrit')

sys.meta_path.insert(0, NoPyrit())
import midojo
from midojo import Episode, grade, open_episode
assert 'pyrit' not in sys.modules
try:
    import midojo.pyrit
except ImportError as error:
    assert 'midojo[pyrit]' in str(error), str(error)
else:
    raise AssertionError('optional import should explain the missing extra')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
