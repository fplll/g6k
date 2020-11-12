Install as you would regular G6K (https://github.com/fplll/g6k). To run the experiments from [1] use

python bkz.py 180 --bkz/alg slide --slide/overlap 1 --bkz/blocksizes 60:61:1 --bkz/pre_blocksize 45 --bkz/tours 120 --challenge_seed 0 --verbose

and 

python bkz.py 170 --bkz/alg slide --slide/overlap 1 --bkz/blocksizes 85:86:1 --bkz/pre_blocksize 60 --bkz/tours 100 --challenge_seed 0  --verbose

for overlap and challenge_seed in [1,5,10,15] and [0,1,2,...,9], respectively.

[1] The Convergence of Slide-type Reduction. Michael Walter. 2020
