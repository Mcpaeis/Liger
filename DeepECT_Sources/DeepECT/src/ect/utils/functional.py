

from itertools import islice

def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def count_values_in_sequence(seq):
    from collections import defaultdict
    res = defaultdict(lambda : 0)
    for key in seq:
        res[key] += 1
    return dict(res)