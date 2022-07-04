import multiprocessing as mp
import numpy as np


n_samples = 1000
n_cores = 1

results = {}

def fun(a, b):
    return a + b
    
def run_bootstrap(results, n_samples, n_cores):

    def main():
        pool = mp.Pool(n_cores)
        args = [(np.random.rand(1), np.random.random(1)) for boot in range(n_samples)]

        bootstrap_results = pool.starmap(fun, args)
        ave = 0.0

        for res in bootstrap_results:
            ave += res

        ave /=  n_samples

    if __name__ == '__main__':  
        results['a'] = main()

run_bootstrap(results, n_samples, n_cores)

print(results)

# run_bootstrap(n_samples, n_cores)
