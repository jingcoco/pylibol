#!D:\SOL-master\pyenv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'sol==1.1.0','console_scripts','sol_train'
__requires__ = 'sol==1.1.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('sol==1.1.0', 'console_scripts', 'sol_train')()
    )
