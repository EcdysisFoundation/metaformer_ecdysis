from logging import DEBUG, INFO, basicConfig

TAXON_LEVELS = levels = ['order', 'family', 'genus']
SEED = 42
LOGGING_LEVEL = INFO

basicConfig(format='%(levelname)s :: %(name)s - %(asctime)s :: %(message)s')
