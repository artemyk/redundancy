# -*- coding: utf-8 -*-
import ppl
from utils import get_best_solution, mi

import numpy as np

# Compute our proposed redundancy measure, $I_\cap^\star$.

# Given a joint distribution p_{Y,X_1,X_2}, our redundancy measure is 
# the solution to the following optimization problem:
# 
# R = min_{s_{Q|Y}} I_s(Y;Q) 
#              s.t. s_{Q|Y} ⪯ p_{X_i|Y} ∀i
# 
# This can in turn be re-written as:
#  R = min_{s_{Q|Y},s_{Q|X_1}, .., s_{Q|X_n}} I_s(Y;Q)
#      s.t. ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)
# 
# Note that this is optimization problem involes a maximization of 
# a convex function subject to a system of linear constraints.  This 
# system of linear constraints defines a convex polytope, and the maximum 
# of the function will lie at one of the vertices of the polytope.  We 
# proceed by finding these vertices, iterating over them, and finding 
# the vertex with the largest value of I(Y;Q).
# We use the computational geometry library 'ppl'.



def get_Istar(raw_pjoint, eps=1e-8):
    # raw_pjoint is a joint distribution object from dit, where the last random
    #    variable is the target Y, and the others are the sources X_1 ,..., X_n
    # eps is needed because we round our conditional probability distributions
    #    to rationals (the library ppl requires rationals)
    
    pjoint = raw_pjoint.copy()
    
    target_rvndx = len(pjoint.rvs) - 1
    pY = pjoint.marginal([target_rvndx,])
    
    if len(pY) == 1:
        # Trivial case where target has a single outcome, redundancy has to be 0
        return 0, {}
    
    # Set up a ppl object that will encode a system of linear inequalities
    cs = ppl.Constraint_System()
    
    # variablesQgiven holds ppl variables that represent conditional probability
    # values of s(Q=q|X_i=x_i) for different sources i, as well as s(Q=q|Y=y)
    # for the target Y (recall that Q is our redundancy random variable)
    variablesQgiven = {} 

    num_vars        = 0  # counter for tracking how many ppl variables we've created
    

    # Calculate the maximum number of outcomes we will require for Q
    n_q = sum([len(alphabet)-1 for rvndx, alphabet in enumerate(pjoint.alphabet)
                             if rvndx != target_rvndx]) + 1
                             
    # Iterate over all the random variables (R.V.s): i.e., all the sources + the target 
    for rvndx, rv in enumerate(pjoint.rvs):
        variablesQgiven[rvndx] = {}

        mP = pjoint.marginal([rvndx,]) # the marginal distribution over the current R.V.
        if len(mP._outcomes_index) != len(pjoint.alphabet[rvndx]):
            raise Exception('All marginals should have full support ' +
            	            '(to proceed, drop outcomes with 0 probability)')

        # Iterate over outcomes of current R.V.
        for v in pjoint.alphabet[rvndx]:
            sum_to_one = 0 
            for q in range(n_q):
                cvar        = ppl.Variable(num_vars)
                num_vars   += 1
                
                # represents s(Q=q|X_rvndx=v) if rvndx != target_rvndx
                #        and s(Q=q|Y=v)       if rvndx == target_rvndx
                variablesQgiven[rvndx][(q, v)] = cvar 

                cs.insert(cvar >= 0) # Enforces non-negativity

                sum_to_one += cvar

            if rvndx != target_rvndx:
            	# Linear constraint that enforces Σ_q s(Q=q|X_rvndx=v) = 1
                cs.insert(sum_to_one == 1)
                
    # Now we add the constraint:
    #    ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)
    for rvndx, rv in enumerate(pjoint.rvs):
        if rvndx == target_rvndx:
            continue

        # Compute joint marginal of target Y and source X_rvndx
        pYSource = pjoint.marginal([rvndx,target_rvndx,], rv_mode='indices')
        for q in range(n_q):
            for y in pjoint.alphabet[target_rvndx]:
                constraint = 0
                cur_mult   = 0.  # multiplier to make everything rational, and make rounded values add up to 1
                for x in pjoint.alphabet[rvndx]:
                	# We divide by eps and round, and then multiply by cur_mult,
                	# to make everything rational
                    pXY        = int( pYSource[pYSource._outcome_ctor((x,y))] / eps )
                    cur_mult   += pXY
                    constraint += pXY * variablesQgiven[rvndx][(q, x)]
                cs.insert(constraint == cur_mult*variablesQgiven[target_rvndx][(q, y)])

    # Define a matrix sol_mx that allows for fast mapping from solution vector 
    # returned by ppl (i.e., a particular extreme point of our polytope) to a 
    # joint distribution over Q and Y
    n_y    = len(pY)
    sol_mx = np.zeros((num_vars, n_q*n_y))
    for (q,y), k in variablesQgiven[target_rvndx].items():
        y_ix = pY._outcomes_index[y]
        sol_mx[k.id(), q*n_y + y_ix] += pY[y]

    def get_solution_val(x, full=False):
    	# Computes the mutual information for a particular solution vector 
    	# return by ppl (in practice, this will be some extreme point of our polytope)
    	# x is the solution vector
    	# full=True returns full solution information.Slower and only used once best solution is found

        # Map solution vector x to joint distribution over Q and Y
        sol = { 'pQY' : x.dot(sol_mx).reshape((n_q,n_y)) }

        if full:
	        # Compute conditional distributions of Q given each source X_i
            # This is not necessary to figure out which extreme point is best
            for rvndx in range(len(pjoint.rvs)-1):
                pX = pjoint.marginal([rvndx,])
                sol[rvndx] = np.zeros( (n_q, len(pX.alphabet[0]) ) )
                for (q,v), k in variablesQgiven[rvndx].items():
                    v_ix = pX._outcomes_index[v]
                    sol[rvndx][q,v_ix] = x[k.id()]

        # Return solution information and mutual information I(Q;Y)
        return sol, mi(sol['pQY'])
    
    # The following uses ppl to turn our system of linear inequalities into a 
    # set of extreme points of the corresponding polytope. It then calls 
    # get_solution_val on each extreme point
    return get_best_solution(cs, get_solution_val)

