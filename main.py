import os
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 이상이 필요합니다. Python 3.12로 가상환경을 생성하세요.")

# Avoid CPU oversubscription in tiny matrix multiplications; override if desired.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

from rushhour.gui import main

if __name__ == '__main__':
    main()
