"""
Misc util. Ported necessary components from
https://github.com/openai/baselines/blob/master/baselines/common/misc_util.py
"""

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
