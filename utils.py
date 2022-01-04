# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import ppl
import pyximport; pyximport.install()
from mi import mi

def point2array(p):
    # Convert a solution vector in ppl format to a return numpy array
    x = np.fromiter(p.coefficients(), dtype='double')
    x = x/float(p.divisor())
    return x

def get_best_solution(cs, get_solution_val):
    # This function maximizes a convex function over a linear polytope.
    # It takes a system of linear inequalities, uses ppl to convert this to
    # a list of extreme points, evaluates get_solution_val on each extreme point,
    # and finally returns the extreme point with the largest get_solution_val.
    # 
    # Paramters:
    #   cs - system of linear inequalities, in ppl format
    #   get_solution_val - function to call on each extreme point

    # convert linear inequalities into a list of extreme points
    poly_from_constraints = ppl.C_Polyhedron(cs)
    all_generators = poly_from_constraints.minimized_generators()

    best_x, best_val, best_sol = None, -np.inf, None
    # iterate over the extreme points
    for gen in all_generators:
        if not gen.is_point():
            # this happens if the linear inequalities specify a cone,
            # rather than a polytope.
            raise Exception('Returned solution not a point: %s'%gen)
            
        # Get extreme point x, and evaluate get_solution_val on it
        x        = point2array(gen)
        sol, val = get_solution_val(x, full=False)

        if val > best_val:
            best_x, best_val, best_sol = x, val, sol
            
    if best_sol is None:
        raise Exception('No solutions found!')

    # Call the 'full' version of get_solution_val on the best 
    # extreme point (this computes additional solution information) 
    best_sol, final_best_val = get_solution_val(best_x, full=True)

    # Sanity check
    assert(np.isclose(final_best_val, best_val))
            
    return best_val, best_sol

