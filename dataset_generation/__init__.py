from logging import DEBUG, INFO, basicConfig

SEED = 42
LOGGING_LEVEL = INFO

basicConfig(format='%(levelname)s :: %(name)s - %(asctime)s :: %(message)s')
