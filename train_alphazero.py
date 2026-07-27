"""Compatibility launcher for the packaged AlphaZero trainer."""

import multiprocessing as mp

from alphazero.training import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
