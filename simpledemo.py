from __future__ import print_function
import dit
import blackwell_redundancy 

pjoint = dit.Distribution(['000','010','100','111'], [0.25, 0.25, 0.25, 0.25]) # simple AND gate
print('Iprec:', blackwell_redundancy.get_Iprec(pjoint)[0])

