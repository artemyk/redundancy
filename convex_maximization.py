"""
Code for maximizing a convex function over a polytope, as defined
by a set of linear equalities and inequalities.

This uses the fact that the maximum of a convex function over a
polytope will be achieved at one of the extreme points of the polytope.

Thus, the maximization is done by taking a system of linear inequalities,
using the pypoman library to create a list of extreme
points, and then evaluating the objective function on each point.

We use various techniques to first prune away redundant constraints.
"""

import numpy as np
import scipy
import pypoman

__all__ = (
    'maximize_convex_function',
)

import scipy.spatial.distance as sd

ABS_TOL = 1e-7

def remove_duplicates(A,b):
    # Removes duplicate rows from system (in)equalities given by A and b
    while True:
        N = len(A)
        mx = np.hstack([b[:,None], A])
        dists = sd.squareform(sd.pdist(mx))
        duplicates_found = False
        for ndx1 in range(N):
            A1, b1 = A[ndx1,:], b[ndx1]
            keep_rows = np.ones(N, bool)
            keep_rows[ndx1+1:] = dists[ndx1,ndx1+1:]>ABS_TOL
                    
            if not np.all(keep_rows):
                duplicates_found = True
                A = A[keep_rows,:]
                b = b[keep_rows]
                break

        if not duplicates_found:
            break 

    return A, b


def eliminate_redundant_constraints(A,b):
    # Eliminate redundant constraints from the inequality system Ax <= b
    init_num_cons = A.shape[0]
    A, b = A.copy(), b.copy()
    
    N = A.shape[1]
    bounds = [(None,None),]*N
    
    keep_rows         = list(range(A.shape[0]))
    nonredundant_rows = set([])
    
    while True:
        eliminated = False
        for i in keep_rows:
            
            if i in nonredundant_rows:  # already tested this row
                continue
                
            # Current row provides the constraint b >= a^T x
            # Let A' and b' indicate all other constraints. 
            # If b >= max_x a^T x  such that b' >= A'x, then this constraint is redundant and can be eliminated

            other_rows = [j for j in keep_rows if j != i]
            c = scipy.optimize.linprog(-A[i], A_ub=A[other_rows], b_ub=b[other_rows], bounds=bounds)
            optval = -c.fun
            if c.status != 0:  # LP solver did not succeed, row is not redundant
                nonredundant_rows.add(i)
                continue
                
            if -c.fun <= b[i] + ABS_TOL:  # redundant row
                keep_rows = other_rows
                eliminated = True
                break

        if not eliminated:
            break
            
    return A[keep_rows], b[keep_rows]


def maximize_convex_function(f, A_ineq, b_ineq, A_eq=None, b_eq=None):
    """
    Maximize a convex function over a polytope.

    Parameters
    ----------
    f : function
        Objective function to maximize
    A_ineq : matrix
        Specifies inequalities matrix, should be num_inequalities x num_variables
    b_ineq : array
        Specifies inequalities vector, should be num_inequalities long
    A_eq : matrix
        Specifies equalities matrix, should be num_equalities x num_variables
    b_eq : array
        Specifies equalities vector, should be num_equalities long

    Returns tuple optimal_extreme_point, maximum_function_value

    """

    best_x, best_val = None, -np.inf
    
    A_ineq = A_ineq.astype('float')
    b_ineq = b_ineq.astype('float')
    A_ineq, b_ineq = remove_duplicates(A_ineq, b_ineq)

    if A_eq is not None:
        # pypoman doesn't support equality constraints. We remove equality 
        # constraints by doing a coordinate transformation.

        A_eq = A_eq.astype('float')
        b_eq = b_eq.astype('float')
        A_eq, b_eq = remove_duplicates(A_eq, b_eq)

        # Get one solution that satisfies A x0 = b
        x0 = scipy.linalg.lstsq(A_eq, b_eq)[0]
        assert(np.abs(A_eq.dot(x0) - b_eq).max() < 1e-5)

        # Get projector onto null space of A, it satisfies AZ=0 and Z^T Z=I
        Z = scipy.linalg.null_space(A_eq)
        # Now every solution can be written as x = x0 + Zq, since A x = A x0 = b 

        # Inequalities get transformed as
        #   A'x <= b'  to   A'(x0 + Zq) <= b  to  (A'Z)q <= b - A'x0

        b_ineq = b_ineq - A_ineq.dot(x0)
        A_ineq = A_ineq.dot(Z)

        A_ineq, b_ineq = remove_duplicates(A_ineq, b_ineq)
        
        transform = lambda q: Z.dot(q) + x0

    else:
        transform = lambda x: x

    A_ineq, b_ineq = eliminate_redundant_constraints(A_ineq, b_ineq)
    
    extreme_points = pypoman.compute_polytope_vertices(A_ineq, b_ineq)
    
    for v in extreme_points:
        x = transform(v)
        val = f(x)
        if val > best_val:
            best_x, best_val = x, val

    if best_x is None:
        raise Exception('No extreme points found!')

    return best_x, best_val


