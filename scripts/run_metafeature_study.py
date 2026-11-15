import argparse

from iml_interpretability.metafeature_study import run

parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true")
run(pilot=parser.parse_args().pilot)
