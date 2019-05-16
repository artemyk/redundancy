import numpy as np
import ppl

def entropy(probs):
    assert(np.isclose(probs.sum(), 1.))
    assert(np.all(probs>=0.))
    return -sum(p*np.log2(p) for p in np.ravel(probs) if p != 0.)

def mutual_info(pAB):
    return entropy(pAB.sum(axis=0)) + entropy(pAB.sum(axis=1)) - entropy(pAB)

def kl(p,q):
    assert(np.isclose(p.sum(), 1.))
    assert(np.all(p>=0.))
    assert(np.isclose(q.sum(), 1.))
    assert(np.all(q>=0.))
    assert(len(p)==len(q))
    return sum(p[ndx]*np.log2(p[ndx]/q[ndx]) for ndx in range(len(p)) if p[ndx] != 0.)


def point2array(p):
    x = np.array(list(map(float, p.coefficients())))
    x /= float(p.divisor())
    return x

def get_best_solution(cs, get_solution_val):
    poly_from_constraints = ppl.C_Polyhedron(cs)
    all_generators = poly_from_constraints.minimized_generators()


    best_val, best_sol = -np.inf, None
    for gen in all_generators:
        if not gen.is_point():
            raise Exception
            
        x        = point2array(gen)
        sol, val = get_solution_val(x)

        if val > best_val:
            best_val, best_sol = val, sol
            
    return best_val, best_sol

