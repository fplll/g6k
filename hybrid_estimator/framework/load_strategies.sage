# Apparently the internal loader for
# strategies of fplll/fpylll is broken through sage
# Doing that by hand...
import json
from fpylll.fplll.bkz_param import Strategy

with open("../framework/bkz_strat.json") as json_data:
    data = json.load(json_data)

strategies = 91 * [None]

for datum in data:
    b = datum["block_size"]
    prun = datum["pruning_parameters"]
    prep = datum["preprocessing_block_sizes"]
    strat = Strategy(b, prep, prun)
    strategies[b] = strat
