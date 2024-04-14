# Run some tests

from __future__ import print_function
from collections import OrderedDict
import time

import numpy as np
import dit
from blackwell_redundancy import get_Iprec


to_test = []  # add tuples (name, dit distribution, target value to 3 decimal places, opts to pass to get_Iprec)

# simple UNQ gate
to_test.append(('X1 unique gate',
                dit.Distribution(['000','010','101','111'], [0.25, 0.25, 0.25, 0.25]),
                0.0,
                None))

# simple AND gate
to_test.append(('2-way AND gate',
                dit.Distribution(['000','010','100','111'], [0.25, 0.25, 0.25, 0.25]),
                0.311,
                None))

# a few trivariate distributions
trivariate_dists = OrderedDict()
states = []
for i in range(2**3):
    s = format(i, '03b')
    states.append(s + ('0' if s!='111' else '1'))
to_test.append(('Y=X1 AND X2 AND X3', dit.Distribution(states, np.ones(len(states))/len(states)), 0.138, None))
to_test.append(('Y=X1 + X2 + X3'    , dit.pid.distributions.trivariates['sum'], 0.311, None))

# Creates the overlap gate
# X1=(A,B), X2=(A,C), X3=(A,D) (one variable in common)
# Y=(X1,X2,X3)
states = []
ndx = 0
statenames='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@'
for x1 in range(4):
    for x2 in range(4):
        for x3 in range(4):
            x1a = int(x1<2)
            x2a = int(x2<2)
            x3a = int(x3<2)
            if x1a != x2a or x1a != x3a:
                continue
            sname = str(x1)+str(x2)+str(x3)+statenames[ndx]
            states.append(sname)
            ndx+=1 
    
to_test.append(('Y=((A,B),(A,C),(A,D))', dit.Distribution(states, np.ones(len(states))/len(states)), 1., None))


# tricky distribution 

dat = [21, 0, 1, 12, 2, 0, 15, 0, 3, 4, 7, 1, 15, 0, 2, 10, 1, 0, 7, 9, 3, 2, 9, 4, 6, 1, 0, 3, 9,
         4, 2, 9, 7, 0, 4, 8, 0, 9, 7, 0, 5, 5, 0, 2, 8, 0, 0, 8]
pmf = dat / np.sum(dat)
out2 = [ '000', '001',  '002',  '010', '011', '012', '020', '021', '022','030', '031', '032',
         '100', '101', '102', '110', '111', '112','120',  '121', '122', '130', '131' ,'132',
         '200','201',  '202', '210', '211', '212','220', '221', '222', '230', '231', '232',
         '300', '301', '302', '310', '311', '312', '320', '321', '322', '330', '331', '332']

to_test.append(('3x4 n_q=3', dit.Distribution(out2, pmf), 0.110, {'n_q':3}))
to_test.append(('3x4 n_q=4', dit.Distribution(out2, pmf), 0.119, {'n_q':4}))
    

for name, pjoint, tv, opts in to_test:
    start_time = time.time()
    v = np.round(get_Iprec(pjoint, **(opts if opts else {}))[0], 3)
    took_time  = time.time() - start_time
    if not np.isclose(v, tv):
        print("%25s (%5.2fs) | ERROR  : got %0.3f, expected %0.3f"  % (name, took_time, v, tv))
    else:
        print("%25s (%5.2fs) | success: got %0.3f"  % (name, took_time, tv))

    
