import numpy as np
from conllu import parse_incr
from statsmodels.stats.contingency_tables import mcnemar
import networkx as nx
from itertools import combinations

'''-----Input variables-----'''
'''
    The variable setup sets how the script should be run: with or without filtered data (the first character), with or without superpattern subtraction (the second character).
    Setup must be (NN|NY|YN|YY).
    
    lang_a and lang_b correspond to the language abbreviations used in the Data folder.
'''
setup = 'NN'
lang_a, lang_b = sorted(('es', 'fr'))
'''-------------------------'''

PATH_ROOT = '/data/tatoebaAligned/es-fr_small/'

lang_pair = lang_a.split('.')[-1] + '-' + lang_b.split('.')[-1]
A_trans = PATH_ROOT + lang_a + '.ord.trans'
B_trans = PATH_ROOT + lang_b + '.ord.trans'
if setup[0] == 'N':
    A_orig = PATH_ROOT + lang_a + '.conllu'
    B_orig = PATH_ROOT + lang_b + '.conllu'
elif setup[0] == 'Y':
    A_orig = '../Data/conllu/' + lang_pair + '/conllu.filter.' + lang_a
    B_orig = '../Data/conllu/' + lang_pair + '/conllu.filter.' + lang_b


A_B_parallel = zip(parse_incr(open(A_orig, 'r', encoding='utf8')), parse_incr(open(B_orig, 'r', encoding='utf8')))

def subpattern_in_pattern_indexes(a, b, overlap = True):
    '''
    Checks if sequence b is in sequence a, allowing for gaps like SQS (Tatti and Vreeken, 2012).
    The number of gaps (inside b) must be strictly less than the length of sequence b.
    
    The variable overlap sets whether a sequence AB should be counted twice in ABB, in which the element A is re-used.
    
    The function returns a list of matches of pattern b in a, containing the indices of the matching elements.
    E.g., if overlap=True, then
    
    subpattern_in_pattern_indexes('OABB', 'AB', overlap=True)
    >>> [[1, 2], [1, 3]] 
    Kirian said : shoud be the above output, but output like if overlap=False

    subpattern_in_pattern_indexes('OABBAB', 'AB', overlap=False)
    >>> [[1, 2], [4, 5]]
    '''
    C = []
    a = list(a)
    orig_len_a = len(a)
    b = list(b)
    total_used_indices = set()
    while True:
        match = []
        used_indices = set()
        gap_pool = len(b) - 1
        try:
            while a[0] != b[0]:
                a = a[1:]
        except IndexError:
            return C
        i = 0
        j = 0
        while i in range(len(a)) and j in range(len(b)):
            if a[i] == b[j] and i + (orig_len_a - len(a)) not in total_used_indices:
                match.append(i + (orig_len_a - len(a)))
                if j == len(b) - 1:
                    C.append(match)
                    if not overlap:
                        used_indices.add(i + (orig_len_a - len(a)))
                        total_used_indices |= used_indices
                    a = a[1:]
                    break
                j += 1

                if not overlap:
                    used_indices.add(i + (orig_len_a - len(a)))
            elif a[i] != b[j] and gap_pool > 0:
                gap_pool -= 1
            elif a[i] != b[j] and gap_pool == 0:
                a = a[1:]
                break
            i += 1
        else:
            a = a[1:]


# Vocabulary : unigrams of POS
Vocabulary = set()

# Pattern : All patterns in both 
Patterns = set()
for pattern in open(A_trans, 'r'):
    pattern = tuple(pattern.split(' (')[0].split())
    Vocabulary |= set(pattern)
    Patterns.add(pattern)
for pattern in open(B_trans, 'r'):
    pattern = tuple(pattern.split(' (')[0].split())
    Vocabulary |= set(pattern)
    Patterns.add(pattern)
Vocabulary = {(x,) for x in Vocabulary}
Patterns |= Vocabulary


A_list_of_counters = []
B_list_of_counters = []

max_sentence_pairs = None
sentences_counter = 0
for sentence_a, sentence_b in A_B_parallel:
    sentences_counter+=1
    if sentences_counter%100 == 0:
        print(sentences_counter)
    if sentences_counter == max_sentence_pairs:
        break
    sentence_a = [token['upostag'] for token in sentence_a]
    sentence_b = [token['upostag'] for token in sentence_b]
    bop_a = {} #bag of patterns
    bop_b = {}
    for pattern in Patterns:
        indexes_in_a = subpattern_in_pattern_indexes(sentence_a, pattern, overlap=False)
        if len(indexes_in_a) > 0:
            bop_a[pattern] = indexes_in_a
        
        indexes_in_b = subpattern_in_pattern_indexes(sentence_b, pattern, overlap=False)
        if len(indexes_in_b) > 0:
            bop_b[pattern] = indexes_in_b

    # Using networkx graphs to represent Hasse diagrams so as to be able to deal with superpattern subtractions.
    Ga = nx.DiGraph()
    Gb = nx.DiGraph()
    for Gx, sentence_x, bop_x, loc in [(Ga, sentence_a, bop_a, A_list_of_counters), (Gb, sentence_b, bop_b, B_list_of_counters)]:
        root = tuple(range(len(sentence_x)))
        Gx.add_node(root)
        all_nodes = [tuple(match) for p in bop_x.values() for match in p]
        for n in all_nodes:
            if not n == root:
                Gx.add_edge(root, n)
        for x, y in combinations(all_nodes, 2):
            if len(subpattern_in_pattern_indexes(x, y, overlap=False)) > 0:
                Gx.add_edge(x, y)
            elif len(subpattern_in_pattern_indexes(y, x, overlap=False)) > 0:
                Gx.add_edge(y, x)
        Gx = nx.algorithms.dag.transitive_reduction(Gx)
        pattern_ancestors = [(n, nx.algorithms.dag.ancestors(Gx, n)) for n in Gx.nodes]
        bop_x = {}
        for p, ancestors in pattern_ancestors:
            p = tuple(sentence_x[i] for i in p)
            if p not in Patterns:
                continue
            ancestors = {tuple(sentence_x[i] for i in A) for A in ancestors}
            ancestors = {A for A in ancestors if A in Patterns}
            ancestors = frozenset(ancestors)
            bop_x[(p, ancestors)] = bop_x.get((p, ancestors), 0) + 1
        loc.append(bop_x)

print(sentences_counter)
print("Finished reading all sentences.")
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
# This is mapping sets of patterns to ordinal labels 
dv = dv.fit(A_list_of_counters + B_list_of_counters)

# A is a matrix of (N,M); N is the number of sentences and M the number of patterns, so (i,j) will
# represent the number of pattern j in the sentence i 
A = dv.transform(A_list_of_counters)
B = dv.transform(B_list_of_counters)


# print("KK A.shape", A.shape)
# print("KK A", A[0])
# print("KK A", A[0].shape)

# print("KK A_list_of_counters", A_list_of_counters[0])
# print("KK A_list_of_counters", len(A_list_of_counters))
# print(list(dv.vocabulary_.items())[521])

pattern_ancestors = [(i, p) for i, p in enumerate(dv.get_feature_names())]
Patterns = sorted(Patterns, key=len, reverse=True)
print("No. of patterns:", len(Patterns))
print(pattern_ancestors[377])
exit()
found_differences = {}
for pattern in Patterns:
    pattern_indices = [(i, p) for i, p in pattern_ancestors if p[0] == pattern]
    if setup[1] == 'Y':
        pattern_indices = [(i, p) for i, p in pattern_indices if not any(ancestor in found_differences for ancestor in p[-1])]
    if pattern_indices == []:
        continue
    pattern_indices = [i for i, p in pattern_indices]
    counts = np.sum(A[:,pattern_indices], axis=-1) - np.sum(B[:,pattern_indices], axis=-1)
    a = int(np.sum(np.sum(A[:,pattern_indices], axis=-1)[counts == 0]))
    b = int(np.sum(counts[counts > 0]))
    c = int(-np.sum(counts[counts < 0]))
    if b + c == 0:
        chi2 = 0.0
    else:
        chi2 = (b-c)**2 / (b+c)
    print(pattern, ':', (chi2, a, b, c))
    if chi2 >= 3.841: #p=0.05
        found_differences[pattern] = (chi2, b, c)
found_differences = sorted(found_differences.items(), key=lambda I: I[1][0], reverse=False)

print()
output_file = open(PATH_ROOT + lang_pair + '.' + setup + '.tsv', 'w')
output_file.write('\t'.join(['pattern', 'chi2', lang_a, lang_b]) + '\n')
for p in found_differences:
    print(p)
    p = list(p)
    p[0] = ' '.join(p[0])
    p[1] = '\t'.join(str(x) for x in p[1])
    output_file.write('\t'.join(p) + '\n')

print("Found differences:", len(found_differences))