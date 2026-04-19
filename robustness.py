"""
Experiments to check kronmem's sensitivity to noise in values and addresses
TODO:
- ablation study on kronmem activation
- test robustness with *over*writes over time, like krop paper
"""
from time import perf_counter
import pickle as pk
import itertools as it
import torch as tr
import matplotlib.pyplot as pt
from kronmem import KroneckerMemory

def trial(K, L, sigma):

    # sample random indices for addresses and values
    addresses = tr.randperm(2**K)[:L]
    values = tr.randint(2**K, (L,))
    
    # store associations
    km = KroneckerMemory(K)
    M = km.init()
    for (i_a, i_b) in zip(addresses, values):
        # add noise to address and value
        v_a = km.embed(i_a) + sigma * tr.randn(K)
        v_b = km.embed(i_b) + sigma * tr.randn(K)
        # store
        a = km.expand(v_a)
        M = km.write(M, a, v_b)
    
    # read and check associations
    num_correct = 0
    for (i_a, i_b) in zip(addresses, values):
        v_a = km.embed(i_a)
        v_b = km.embed(i_b)
    
        a = km.expand(v_a)
        v = km.read(M, a)
    
        num_correct += (v.sign() == v_b).all()

    return num_correct

if __name__ == "__main__":

    do_trials = False
    show_trials = True

    Ks = tr.arange(3,11).tolist()
    sigmas = tr.linspace(0, .5, 8).tolist()
    num_reps = 10

    # Ks = tr.arange(3,5).tolist()
    # sigmas = tr.linspace(0, .5, 2).tolist()
    # num_reps = 2

    if do_trials:

        retrieval_rates = {}
        trial_times = {}
        for (K, sigma, rep) in it.product(Ks, sigmas, range(num_reps)):
            for L in (2**tr.arange(K+1)).tolist():
                start_time = perf_counter()
                num_correct = trial(K, L, sigma)
                trial_time = perf_counter() - start_time
                print(f"{K=}, {L=}, {sigma=}, {rep=}: retrieval rate = {num_correct / L} in {trial_time:.3f}s")

                retrieval_rates[K, sigma, rep, L] = num_correct / L
                trial_times[K, sigma, rep, L] = trial_time

            with open("robustness.pkl","wb") as f:
                pk.dump((retrieval_rates, trial_times), f)

    if show_trials:
        with open("robustness.pkl","rb") as f:
            (retrieval_rates, trial_times) = pk.load(f)

        pt.figure(figsize=(16,8))
        for sp, K in enumerate(Ks):
            pt.subplot(2, 4, sp+1)
            Ls = (2**tr.arange(K+1)).tolist()
            for c, L in enumerate(Ls):
                color = (.5 * c / len(Ls),)*3
                rates = tr.zeros(num_reps, len(sigmas))
                for (rep, (s, sigma)) in it.product(range(num_reps), enumerate(sigmas)):
                    rates[rep,s] = retrieval_rates[K, sigma, rep, L]
                pt.plot(sigmas, rates.t(), '.', color=color)
                pt.plot(sigmas, rates.mean(dim=0), '-', color=color, label=f"{L=}")

            pt.ylim([-.1, 1.1])
            pt.title(f"{K=}")
            pt.legend()
        pt.gcf().supxlabel("Noise Standard Deviation")
        pt.gcf().supylabel("Retrieval Rate")
        pt.tight_layout()
        pt.savefig("retrieval_rate.pdf")
        pt.show()

        pt.figure(figsize=(16,6))
        for sp, K in enumerate(Ks):
            pt.subplot(2, 4, sp+1)
            Ls = (2**tr.arange(K+1)).tolist()
            tolerance = tr.zeros(len(Ls))
            for c,L in enumerate(Ls):
                rates = tr.zeros(num_reps, len(sigmas))
                for (rep, (s, sigma)) in it.product(range(num_reps), enumerate(sigmas)):
                    rates[rep,s] = retrieval_rates[K, sigma, rep, L]
                rates = rates.mean(dim=0)
                if rates[0] == 1.0:
                    tolerance[c] = sigmas[(rates != 1.0).numpy().argmax()-1]
            pt.plot(Ls, tolerance, 'ko-')
            pt.ylim([min(sigmas)-.1, max(sigmas)+.1])
            pt.title(f"{K=}")
        pt.gcf().supxlabel("Storage Load")
        pt.gcf().supylabel("Noise Tolerance")
        pt.tight_layout()
        pt.savefig("tolerance.pdf")
        pt.show()

