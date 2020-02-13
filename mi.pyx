# cython: language_level=3

# Cython-optimized mutual information computation
#  (we call this a lot, so it needs to be fast)

cimport cython
from libc.math cimport log
from libc.stdlib cimport malloc, free


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def mi(double[:, :]  pjoint):
    # Given joint distribution pjoint over two random variables Q and Y, 
    # return the mutual information I(Q;Y) in bits

    cdef int n_q = pjoint.shape[0]
    cdef int n_y = pjoint.shape[1]
    cdef double p_q = 0
    cdef double mi = 0
    
    cdef double *p_ys = <double *> malloc(n_y * sizeof(double))
    
    for y in range(n_y):
        p_ys[y] = 0
        
    for q in range(n_q):
        p_q = 0
        for y in range(n_y):
            p = pjoint[q,y]
            if p > 0:
                mi += p*log(p)
                p_q     += p
                p_ys[y] += p
        if p_q > 0:
            mi -= p_q*log(p_q)
        
    for y in range(n_y):
        p = p_ys[y]
        if p > 0:
            mi -= p_ys[y]*log(p)
        
    return mi / log(2.)
