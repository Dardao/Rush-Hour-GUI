import os

# Avoid CPU oversubscription in tiny matrix multiplications; override if desired.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

from rushhour.gui import main

if __name__ == '__main__':
    main()
