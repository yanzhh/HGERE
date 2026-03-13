import logging
import sys
from pathlib import Path


def get_logger(args, log_path, test: bool = False):
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    logger = logging.getLogger()
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)
    if test:
        log_file = f"test_{args.hostname}.log"
    else:
        log_file = f"all_{args.hostname}.log"
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    log_level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    logger.setLevel(log_level)

    # not used: datefmt="%m/%d/%Y %H:%M:%S",
    return logger
