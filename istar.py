import ppl
from utils import get_best_solution, mi

import numpy as np

# Compute our proposed redundancy measure, $I_\cap^\star$

# Given a joint distribution $p_{YX_1X_2}$, $I_\cap^\star$ is 
# written as the solution to the following optimization problem:
# 
# I_alpha = \min_{s_{Q|Y}} \; & I_s(Y;Q)\\
#              s.t. s_{Q|Y} \preceq p_{X_i|Y} \forall i
# 
# This can in turn be re-written as:
# I_\alpha = \min_{s_{Q|Y},s_{Q|X_1}, ..., s_{Q|X_n}}  I_s(Y;Q)\\
#              s.t. \forall i,q,y : \sum_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y).
# 
# Note that this is optimization problem involes a maximization of 
# a convex function, subject to a system of linear constraints.  This 
# system of linear constraints defines a convex polytope, and the maximum 
# of the function will lie at one of the vertices of the polytope.  We 
# proceed by finding these vertices, iterating over them, and finding 
# the vertex with the largest value of I(Y;Q)



def get_Istar(raw_pjoint, eps=1e-8):
    # pjoint is a joint distribution object from dit, where the last random
    #    variable is the target Y, and the others are the sources X_1 ,..., X_n
    # eps is needed because we round our conditional probability distributions
    #    to rationals (computational algebra library ppl only works for rationals)
    
    pjoint = raw_pjoint.copy()
    
    target_rvndx = len(pjoint.rvs) - 1
    
    if len(pjoint.marginal([target_rvndx,])) == 1:
        # Trivial case where target only has a single output
        return 0, {}
    
    cs = ppl.Constraint_System()
    
    variablesQgiven = {}
    num_vars        = 0
    
    n_q = sum([len(alphabet)-1 for rvndx, alphabet in enumerate(pjoint.alphabet)
                             if rvndx != target_rvndx]) + 1
    pY = pjoint.marginal([target_rvndx,])
                             
    for rvndx, rv in enumerate(pjoint.rvs):
        variablesQgiven[rvndx] = {}
        mP = pjoint.marginal([rvndx,])
        if len(mP._outcomes_index) != len(pjoint.alphabet[rvndx]):
            raise Exception('All marginals should have full support (to proceed, drop outcomes with 0 probability)')
        for v in pjoint.alphabet[rvndx]:
            sum_to_one = 0
            for q in range(n_q):
                cvar        = ppl.Variable(num_vars)
                num_vars   += 1
                variablesQgiven[rvndx][(q, v)] = cvar
                sum_to_one += cvar
                cs.insert(cvar >= 0)
            if rvndx != target_rvndx:
                cs.insert(sum_to_one == 1)
                
                
    for rvndx, rv in enumerate(pjoint.rvs):
        if rvndx == target_rvndx:
            continue
        pYSource = pjoint.marginal([rvndx,target_rvndx,], rv_mode='indices')
        for q in range(n_q):
            for y in pjoint.alphabet[target_rvndx]:
                constraint = 0
                cur_mult   = 0.  # multiplier to make everything rational, and make rounded values add up to 1
                for x in pjoint.alphabet[rvndx]:
                    pXY        = int( pYSource[pYSource._outcome_ctor((x,y))] / eps )
                    cur_mult   += pXY
                    constraint += pXY * variablesQgiven[rvndx][(q, x)]
                cs.insert(constraint == cur_mult*variablesQgiven[target_rvndx][(q, y)])

    # Define a matrix that allows for fast mapping from return solution vector to joint distribution over Q and Y
    n_y    = len(pY)
    sol_mx = np.zeros((num_vars, n_q*n_y))
    for (q,y), k in variablesQgiven[target_rvndx].items():
        y_ix = pY._outcomes_index[y]
        sol_mx[k.id(), q*n_y + y_ix] += pY[y]

    def get_solution_val(x, full=False):
        # full=True returns full solution information. It is slower and done for the best solution
        sol = { 'pQY' : x.dot(sol_mx).reshape((n_q,n_y)) }

        if full:
            for rvndx in range(len(pjoint.rvs)-1):
                pX = pjoint.marginal([rvndx,])
                sol[rvndx] = np.zeros( (n_q, len(pX.alphabet[0]) ) )
                for (q,v), k in variablesQgiven[rvndx].items():
                    v_ix = pX._outcomes_index[v]
                    sol[rvndx][q,v_ix] = x[k.id()]

        return sol, mi(sol['pQY'])
    
    return get_best_solution(cs, get_solution_val)

