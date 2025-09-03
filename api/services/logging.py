# Von Chat
import logging, sys, time

def setup_logging(level: str = "INFO"):
    """Einfaches Console-Logging, z. B. [INFO] Nachricht"""
    logger = logging.getLogger()
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)

def set_level(level: str):
    logging.getLogger().setLevel(level.upper())

class Timer():
    """Stopwatch fürs Kontext-Logging.
       Nutzung: with Timer('[RAG] Retrieval'): ..."""
    def __init__(self, label: str, level: int = logging.INFO):
        self.label = label
        self.level = level
        self.t0 = None
        self.ms = 0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ms = int((time.time() - self.t0) * 1000)
        logging.log(self.level, f"{self.label} in {self.ms} ms")
