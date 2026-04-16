import logging
from src.data_loader import FinancialDataLoader


def setup_global_logger():

    logging.basicConfig(
        level=logging.INFO, 
        filename = "log.log", 
        filemode = "w", 
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    setup_global_logger()
    logging.info("Setting up smartComps Engine...")


if __name__ == "__main__":
    main()