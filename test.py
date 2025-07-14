import itertools


def generate_without_duplicates(inputs):
    seen = set()
    print(list(itertools.combinations(range(len(inputs)),2)))
    print(pairs_list)
    for list in inputs:
        pairs_list.append
        
    for prod in itertools.product(inputs):
        prod_set = frozenset(prod)
        if len(prod_set) == 1:  # all items are the same
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod


a = [0, 1]
b = [2, 3, 4]
c = [5,7,8,9]

for val in generate_without_duplicates([a, b,c]):
    print(val)
