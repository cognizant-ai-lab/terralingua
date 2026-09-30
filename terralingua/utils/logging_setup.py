import logging

_VERBOSE_TO_LEVEL = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

_NOISY_LIBS = ("litellm", "httpx", "anthropic", "openai", "urllib3", "httpcore")


def setup_logging(verbose: int = 1) -> None:
    level = _VERBOSE_TO_LEVEL.get(verbose, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)


def logger_level(verbose: int) -> int:
    return _VERBOSE_TO_LEVEL.get(verbose, logging.INFO)
