import numpy as np
import ppl
import pyximport; pyximport.install()
from mi import mi

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
            
    if best_sol is None:
        raise Exception('No solutions found!')
            
    return best_val, best_sol

