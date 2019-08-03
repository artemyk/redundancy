import ppl
from utils import mutual_info, get_best_solution
import numpy as np

# Finding I_star

# Given a joint distribution $p_{YX_1X_2}$, $I_\cap^\star$ is 
# written as the solution to the following optimization problem:
# 
# I_alpha = \min_{s_{Q|Y}} \; & I_s(Y;Q)\\
#              s.t. s_{Q|Y} \preceq p_{X_i|Y} \forall i
# 
# This can in turn be re-written as:
# I_\alpha = \min_{s_{Q|Y},s_{Q|X_1}, ..., s_{Q|X_n}}  I_s(Y;Q)\\
#              s.t. \forall i,q,y : \sum_{x_i} s(q|x_i) p(x_i|y) = s(q|y).
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
    ndx             = 0
    
    n_q = sum([len(alphabet)-1 for rvndx, alphabet in enumerate(pjoint.alphabet)
                             if rvndx != target_rvndx]) + 1
                             
    for rvndx, rv in enumerate(pjoint.rvs):
        variablesQgiven[rvndx] = {}
        mP = pjoint.marginal([rvndx,])
        if len(mP._outcomes_index) != len(pjoint.alphabet[rvndx]):
            raise Exception('All marginals should have full support')
        for v in pjoint.alphabet[rvndx]:
            sum_to_one = 0
            for q in range(n_q):
                cvar        = ppl.Variable(ndx)
                ndx        += 1
                variablesQgiven[rvndx][(q, v)] = cvar
                sum_to_one += cvar
                cs.insert(cvar >= 0)
            if rvndx != target_rvndx:
                cs.insert(sum_to_one == 1)

    for rvndx, rv in enumerate(pjoint.rvs):
        if rvndx == target_rvndx:
            continue
        pY, pSourceGY = pjoint.condition_on(crvs=[target_rvndx,], rvs=[rvndx,], rv_mode='indices')
        for q in range(n_q):
            for y in pjoint.alphabet[target_rvndx]:
                y_ix       = pY._outcomes_index[y]
                constraint = 0
                cur_mult   = 0.  # multiplier to make everything rational, and make rounded values add up to 1
                for x in pjoint.alphabet[rvndx]:
                    pXgY        = int(pSourceGY[y_ix][x] / eps)
                    cur_mult   += pXgY
                    constraint += pXgY * variablesQgiven[rvndx][(q, x)]
                cs.insert(constraint == cur_mult*variablesQgiven[target_rvndx][(q, y)])

    def get_solution_val(x):
        pY  = pjoint.marginal([target_rvndx,])
        sol = {}
        sol['pQY'] = np.zeros( (n_q, len(pY)) )
        for (q,y), k in variablesQgiven[target_rvndx].items():
            y_ix = pY._outcomes_index[y]
            sol['pQY'][q, y_ix] += pY[y] * x[k.id()]

        for rvndx in range(len(pjoint.rvs)-1):
            pX = pjoint.marginal([rvndx,])
            sol[rvndx] = np.zeros( (n_q, len(pX.alphabet[0]) ) )
            for (q,v), k in variablesQgiven[rvndx].items():
                v_ix = pX._outcomes_index[v]
                sol[rvndx][q,v_ix] = x[k.id()]

        return sol, mutual_info(sol['pQY'])
    
    return get_best_solution(cs, get_solution_val)


