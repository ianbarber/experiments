"""Stage 2 task: offset recall and its content-recall twin, embedded in a random token stream.

  offset   [marker_D] [answer]         answer = token exactly D positions before the marker
  content  [cmarker] [key] [answer]    key occurs exactly once earlier, D positions before its copy;
                                       answer = the token that followed it

Stream tokens are uniform over 256 words; keys come from a separate 512-word vocabulary and are unique in the
sequence; offset targets never fall on an original key, so a key occurs exactly once before its query. Content
queries need D >= 4 (at D = 2 the answer slot would be the marker). Only answer tokens are scored.
"""

import numpy as np

T = 2048
DS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
N_PER_TYPE = 55


class Vocab:
    def __init__(self, path="../data/vocab.npz"):
        v = np.load(path)["spaced"]
        self.stream = v[:256]
        self.keys = v[256:768]
        self.markers = v[768:778]           # one per D
        self.cmarker = v[778]


def make_sequence(rng, vocab):
    """Returns ids [T], and per query: answer position, type (0 offset / 1 content), D index."""
    seq = vocab.stream[rng.integers(0, 256, size=T)].copy()
    used = np.zeros(T, dtype=bool)          # positions taken by query tokens
    is_key = np.zeros(T, dtype=bool)        # original key positions (and the answer slot after them is plain)
    queries = []                            # (pos, type, D index)
    keys = rng.permutation(vocab.keys)
    n_keys = 0
    order = [(t, di) for t in (0, 1) for di in rng.integers(0, len(DS), size=N_PER_TYPE)]
    rng.shuffle(order)
    # content queries first, so that key positions are known before offset targets are chosen
    order.sort(key=lambda x: -x[0])
    for typ, di in order:
        D = DS[di]
        L = 3 if typ else 2
        for _ in range(20):
            p = int(rng.integers(D, T - L + 1))           # marker position; needs p - D >= 0
            if used[p:p + L].any() or is_key[p:p + L].any():
                continue
            if typ:
                t = p + 1 - D                             # original key position, D before its copy at p + 1
                if t < 0 or t + 1 >= p or used[t:t + 2].any() or is_key[t:t + 2].any() or (t > 0 and is_key[t - 1]):
                    continue
                seq[t] = keys[n_keys]
                is_key[t] = True
                seq[p], seq[p + 1] = vocab.cmarker, keys[n_keys]
                n_keys += 1
            else:
                if is_key[p - D]:
                    continue
                seq[p] = vocab.markers[di]
            used[p:p + L] = True
            queries.append((p, typ, di))
            break
    # fill answers left to right on the final sequence (a target may itself be an earlier query token)
    queries.sort()
    meta = []
    for p, typ, di in queries:
        D = DS[di]
        if typ:
            seq[p + 2] = seq[p + 1 - D + 1]
            meta.append((p + 2, 1, di))
        else:
            seq[p + 1] = seq[p - D]
            meta.append((p + 1, 0, di))
    return seq, np.array(meta)


def make_batch(rng, vocab, B):
    seqs, metas = zip(*(make_sequence(rng, vocab) for _ in range(B)))
    ids = np.stack(seqs)
    b = np.concatenate([np.full(len(m), i) for i, m in enumerate(metas)])
    m = np.concatenate(metas)
    return ids, b, m                         # m columns: answer position, type, D index


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    v = Vocab()
    rng = np.random.default_rng(0)
    ids, b, m = make_batch(rng, v, 64)
    print("queries per sequence", len(m) / 64, "by type", np.bincount(m[:, 1]) / 64)
    print("per D (offset)", np.bincount(m[m[:, 1] == 0, 2], minlength=10) / 64)
    print("per D (content)", np.bincount(m[m[:, 1] == 1, 2], minlength=10) / 64)
    # verify the definition on the final sequences
    for i in range(64):
        s = ids[i]
        for pos, typ, di in m[b == i]:
            D = DS[di]
            if typ == 0:
                assert s[pos - 1] == v.markers[di] and s[pos] == s[pos - 1 - D]
            else:
                key = s[pos - 1]
                occ = np.where(s[:pos - 1] == key)[0]
                assert s[pos - 2] == v.cmarker and len(occ) == 1 and (pos - 1) - occ[0] == D and s[pos] == s[occ[0] + 1]
    print("definition checks pass")
    s = ids[0]
    p = m[b == 0][5][0]
    print(repr(tok.decode(s[p - 12:p + 3])))
