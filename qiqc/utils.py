import shutil
import sys
from pathlib import Path

import prompter


def rmtree_after_confirmation(path, force=False):
    if Path(path).exists():
        if not force and not prompter.yesno('Overwrite %s?' % path):
            sys.exit(0)
        else:
            shutil.rmtree(path)
