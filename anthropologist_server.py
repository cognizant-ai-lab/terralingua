"""Start the live anthropologist. The code lives in terralingua.anthropologist.server; this file is a shim.

    python anthropologist_server.py --exp_name my_run   is the same as   terralingua-anthropologist --exp_name my_run
"""

from terralingua.anthropologist.server import main

if __name__ == "__main__":
    main()
