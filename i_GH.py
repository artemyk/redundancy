# -*- coding: utf-8 -*-
r"""
Finding I_\cap^GH, as defined in Griffith and Ho (2015)
  (there it is called I_alpha)

Given a joint distribution p_{Y,X_1,..,X_n}, I_\cap^GH can be 
written as the solution to the following optimization problem:
  R = max_{s_{Q|X_1,...,X_n,Y}} I_s(Y;Q)
                       s.t. H(Q|X_i) = H(Q|X_i, Y) ∀i
This can in turn be re-written as:
  R = max_{s_{Q|X_1,...,X_n,Y}} I_s(Y;Q)
       s.t. ∀i,q,x_i,y  s(q|x_i) = s(q|x_i, y)

Note that this is optimization problem involes a maximization of 
a convex function, subject to a system of linear constraints.  This 
system of linear constraints defines a convex polytope, and the maximum 
of the function will lie at one of the vertices of the polytope.  We 
proceed by finding these vertices, iterating over them, and finding 
the vertex with the largest value of I(Y;Q)

Note that, as currently implemented, this can be very slow. Note also that 
we have not derived a bound on the necessary cardinality for Q (though this 
can likely be done, using a similar technique as the bound for Istar)
"""

from scipy.special import entr
import convex_maximization
import numpy as np



def get_I_GH(raw_pjoint, n_q, eps=1e-8):
    """
    Parameters
    ----------
    raw_pjoint: dit distribution
        joint distribution object from dit, where the last random
        variable is the target Y, and the others are the sources X_1 ,..., X_n

    eps: float
        Rounding error. We must round our conditional probability distributions
        to rationals (the library ppl requires rationals)

    Returns 
    -------
    optimum_value : float
        Redundancy value
    sol : dict
        Solution information, in terms of joint distribution p(Q,Y) and 
        conditional distributions p(Q|X_i)
    """    

    pjoint = raw_pjoint.copy()
    
    target_rvndx = len(pjoint.rvs) - 1
    pY     = pjoint.marginal([target_rvndx,])

    if len(pY) == 1:
        # Trivial case where target only has a single output
        return 0, {}
    
    n_sources = len(pjoint.rvs)-1
    rv_names = ['X%d'%(i+1) for i in range(n_sources)] + ['Y',]
    pjoint.set_rv_names(rv_names)

    
    n_y  = len(pjoint.alphabet[-1])
    variables = {}
    for o in pjoint.sample_space():
        for q in range(n_q):
            variables[(q, o)] = len(variables)  # s(q|x_1,...,x_n,y')



    # We represent our feasible set in terms of equalities and inequalities
    A_eq  , b_eq   = [], []  # linear constraints Ax  = b
    A_ineq, b_ineq = [], []  # linear constraints Ax <= b


    # The following puts in place the basic system of constraints:
    # each probability should be non-negative, and the conditional 
    # probabilities should add up to 1.

    # all variables should be positive

    num_vars = len(variables)
    for k,v in variables.items():
        z = np.zeros(num_vars, dtype='int')
        z[v] = -1
        A_ineq.append(z)
        b_ineq.append(0)


    for ndx, o in enumerate(pjoint.sample_space()):
        # for each x1, .., x_n, y : \sum_q s(q|x1,...,x_n,y) = 1 
        sum_to_one = np.zeros(num_vars, dtype='int')
        for q in range(n_q):
            sum_to_one[variables[(q,o)]]=1
        A_eq.append(sum_to_one)
        b_eq.append(1)

        # We now specify the remaining constraints:
        # ∀ q,x_1,y : 
        #   s(q|x_1) = s(q|x_1, y)
        # which is equivalent to
        #   Σ_{x_2,...x_n,y'} s(q|x_1,...,x_n,y') p(x_2,y'|x_1) = \sum_{x_2,...x_n} s(q|x_1,...,x_n,y) p(x_2,...,x_n|x_1,y) ,
        # and similarly for the other sources
        
        def k(d):
            return "".join(d[rv] for rv in rv_names)
        
        for source in pjoint.get_rv_names()[:-1]:
            pX , pOtherYgX = pjoint.condition_on([source], rv_mode='names')
            pXY, pOthergXY = pjoint.condition_on([source, 'Y',], rv_mode='names')
            pXYoutcomes = list(pXY.sample_space())

            # ∀i,q,x_i,y s(q|x_i) = s(q|x_i, y)
            
            for q in range(n_q-1):
                # we can drop constraints on final q due to conservation of probability
                for xy in pXYoutcomes:
                    if pXY[xy] == 0.:
                        continue

                    lhs = np.zeros(num_vars, dtype='int')
                    rhs = np.zeros(num_vars, dtype='int')

                    normL, normR = 0, 0
                    x,y      = xy
                    ix       = pX._outcomes_index[x]        # condition on X
                    otherrvs = pOtherYgX[ix].get_rv_names() # names of RVs on left side of conditioning bar
                    
                    # calculate s(q|x_1) = Σ s(q|x_1,...x_n, y) p(x_2,...x_n, y|x_1)
                    for othervals in pOtherYgX[ix]:
                        p  = pOtherYgX[ix][othervals] # p(X_V\X_i,yy|X_i)
                        valdict = {source: x}
                        valdict.update(dict(zip(otherrvs, othervals)))

                        normL += int(p/eps)
                        lhs[variables[q,k(valdict)]] += int(p/eps)

                    ix       = pXY._outcomes_index[xy]         # condition on X
                    # calculate s(q|x_1,y) = Σ s(q|x_1,...x_n, y) p(x_2,...x_n|x_1,y)
                    otherrvs = pOthergXY[ix].get_rv_names()    # names of RVs on left side of conditioning bar
                    for othervals in pOthergXY[ix]:
                        p  = pOthergXY[ix][othervals] # p(X_V\X_i|X_i,yy)
                        valdict = {source: x, 'Y': y}
                        valdict.update(dict(zip(otherrvs, othervals)))

                        normR += int(p/eps)
                        rhs[variables[q,k(valdict)]] += int(p/eps)

                    A_eq.append(lhs*normR - rhs*normL)
                    b_eq.append(0)
                

    mul_mx = np.zeros( (len(variables), n_q*n_y) )
    for (q,o), k in variables.items():
        y= o[-1]
        y_ix = pY._outcomes_index[y]
        mul_mx[k, q*n_y + y_ix] = pjoint[o]

    probs_y = np.zeros(n_y)
    for y in pjoint.alphabet[target_rvndx]:
        probs_y[pY._outcomes_index[y]] = pY[y]

    H_Y = entr(probs_y).sum()
    ln2 = np.log(2)
    def objective(x):
        # Map solution vector x to joint distribution over Q and Y
        pQY = x.dot(mul_mx).reshape((n_q,n_y))
        probs_q = pQY.sum(axis=1) + 1e-18
        H_YgQ = entr(pQY/probs_q[:,None]).sum(axis=1).dot(probs_q)
        return (H_Y-H_YgQ)/ln2

    # The following uses ppl to turn our system of linear inequalities into a 
    # set of extreme points of the corresponding polytope. It then calls 
    # get_solution_val on each extreme point
    x_opt, v_opt = convex_maximization.maximize_convex_function(
        f=objective,
        A_eq=np.array(A_eq, dtype='int'), 
        b_eq=np.array(b_eq, dtype='int'), 
        A_ineq=np.array(A_ineq, dtype='int'), 
        b_ineq=np.array(b_ineq, dtype='int'))

    sol = {}
    sol['p(Q,Y)'] = x_opt.dot(mul_mx).reshape((n_q,n_y))

    return v_opt, sol   


