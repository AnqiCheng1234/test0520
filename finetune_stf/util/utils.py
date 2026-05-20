import logging
import os


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


__all__ = ["init_log"]
