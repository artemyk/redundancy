from ppl import Variable, Constraint_System
from utils import get_best_solution, mi
import numpy as np
from scipy.stats import entropy

ln2 = np.log(2)

# Finding I_\cap^GH, as defined in Griffith and Ho (2015)
#   (there it is called I_alpha)

# Given a joint distribution $p_{Y,X_1,..,X_n}$, I_\cap^GH can be 
# written as the solution to the following optimization problem:
#   R = max_{s_{Q|X_1,X_2,Y}} I_s(Y;Q)
#              s.t. H(Q|X_i) = H(Q|X_i , Y) \forall i
# This can in turn be re-written as:
#   R = max_{s_{Q|X_1,X_2,Y}} I_s(Y;Q)
#               s.t. \forall i, q, x_i, y  s(q|x_i) = s(q|x_i, y)
# 
# Note that this is optimization problem involes a maximization of 
# a convex function, subject to a system of linear constraints.  This 
# system of linear constraints defines a convex polytope, and the maximum 
# of the function will lie at one of the vertices of the polytope.  We 
# proceed by finding these vertices, iterating over them, and finding 
# the vertex with the largest value of I(Y;Q)

# Note that, as currently implemented, this can be *very* slow. Note also that 
# we have not derived a bound on the necessary cardinality for Q (though this 
# can be done, using a similar technique as the bound for Istar)


def get_I_GH(raw_pjoint, n_q, eps=1e-10):
    # mult is a multiplier that is used to turn floating point numbers into integers,
    # (since ppl only works with rationals)

    pjoint = raw_pjoint.copy()
    
    target_rvndx = len(pjoint.rvs) - 1
    
    if len(pjoint.marginal([target_rvndx,])) == 1:
        # Trivial case where target only has a single output
        return 0, {}
    
    n_sources = len(pjoint.rvs)-1
    rv_names = ['X%d'%(i+1) for i in range(n_sources)] + ['Y',]
    pjoint.set_rv_names(rv_names)
    #print(pjoint.rvs)
    #adsf
    
    n_y  = len(pjoint.alphabet[-1])
    variables = {}
    for o in pjoint.sample_space():
        for q in range(n_q):
            variables[(q, o)] = Variable(len(variables))

    # We represent our feasible set in terms of equalities and inequalities

    # The following puts in place the basic system of constraints:
    # each probability should be non-negative, and the conditional 
    # probabilities should add up to 1.

    # all variables should be positive

    cs = Constraint_System()

    for v in variables.values():
        cs.insert(v >= 0)


    # for each x1, x2, y : \sum_q p(q|x1,x2,y) = 1 
    for ndx, o in enumerate(pjoint.sample_space()):
        cs.insert(sum(variables[(q,o)] for q in range(n_q)) == 1)
        
        # We now specify the remaining constraints:
        # \forall q,x_1,y : 
        #   \sum_{x_2,...x_n,y'} s(q|x_1,...,x_n,y') p(x_2,y'|x_1) = \sum_{x_2,...x_n} s(q|x_1,...,x_n,y) p(x_2,...,x_n|x_1,y) ,
        # and similarly for the other sources
        
        def k(d):
            return "".join(d[rv] for rv in rv_names)
        
        for source in pjoint.get_rv_names()[:-1]:
            pX , pOtherYgX = pjoint.condition_on([source], rv_mode='names')
            pXY, pOthergXY = pjoint.condition_on([source, 'Y',], rv_mode='names')
            pXYoutcomes = list(pXY.sample_space())

            # \forall i, q, x_i, y  s(q|x_i) = s(q|x_i, y)
            
            for q in range(n_q):
                for xy in pXYoutcomes:
                    if pXY[xy] == 0.:
                        continue

                    sumL, sumR = 0, 0
                    x,y      = xy
                    ix       = pX._outcomes_index[x]        # condition on X
                    otherrvs = pOtherYgX[ix].get_rv_names() # names of RVs on left side of conditioning bar
                    
                    # calculate s(q|x_1) = sum s(q|x_1,...x_n, y) p(x_2,...x_n, y|x_1)
                    for othervals in pOtherYgX[ix]:
                        p  = pOtherYgX[ix][othervals] # p(X_V\X_i,yy|X_i)
                        valdict = {source: x}
                        valdict.update(dict(zip(otherrvs, othervals)))

                        sumL += int(p/eps) * variables[q,k(valdict)]

                    ix       = pXY._outcomes_index[xy]         # condition on X
                    # calculate s(q|x_1,y) = sum s(q|x_1,...x_n, y) p(x_2,...x_n|x_1,y)
                    otherrvs = pOthergXY[ix].get_rv_names()    # names of RVs on left side of conditioning bar
                    for othervals in pOthergXY[ix]:
                        p  = pOthergXY[ix][othervals] # p(X_V\X_i|X_i,yy)
                        valdict = {source: x, 'Y': y}
                        valdict.update(dict(zip(otherrvs, othervals)))
                        sumR += int(p/eps) * variables[q,k(valdict)]

                    cs.insert(sumL - sumR == 0)
                

    sol_mx = np.zeros( (n_q*n_y, len(variables)) )
    for (q,o), k in variables.items():
        y=int(o[-1])
        sol_mx[q*n_y + y, k.id()] = pjoint[o]
                             
    def get_solution_val(x):
        pQY = sol_mx.dot(x).reshape(n_q, n_y)
        mival = mi(pQY)
        # pQY = np.zeros((n_q, n_y))
        # for (q,o), k in variables.items():
        #     y=int(o[-1])
        #     pQY[q, y] += pjoint[o] * x[k.id()]
        # mival = entropy(pQY.sum(axis=0)) + entropy(pQY.sum(axis=1)) - entropy(pQY.flat)
        return {'raw':np.array(x), 'pQY':pQY}, mival / ln2
    
    return get_best_solution(cs, get_solution_val)



