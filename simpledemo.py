from __future__ import print_function
import dit
import istar

pjoint = dit.Distribution(['000','010','100','111'], [0.25, 0.25, 0.25, 0.25]) # simple AND gate
print('Istar:', istar.get_Istar(pjoint, n_q=2)[0])

