"""Start the API server. The code lives in terralingua.server.cli; this file is a shim.

    python api_server.py [--host HOST] [--port PORT] [--workers N]   is the same as   terralingua-dashboard ...
"""

from terralingua.server.cli import main

if __name__ == "__main__":
    main()
