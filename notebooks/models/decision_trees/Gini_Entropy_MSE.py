import numpy

# Gini / Entropy calculations =====================================================================
def construct_dict_count(iter_items):
    dict_counts = {}
    for item in iter_items:
        # get(key value, default value if not present)
        dict_counts[item] = dict_counts.get(item, 0) + 1

    return dict_counts

def calculate_gini(list_elements):
    """
    Gini = 1 - sum( count_i**2 / sum(count_i)**2 ) over all i
    """
    if len(list_elements)==0:
        return 1

    iter_counts = construct_dict_count(list_elements).values()
    n = sum(iter_counts)

    array_terms = numpy.array([p_i**2 for p_i in iter_counts])
    array_terms_normalized = numpy.divide(array_terms, n**2) # n can be 0
    scalar_sum = numpy.sum(array_terms_normalized)

    return 1 - scalar_sum

def calculate_entropy(list_elements):
    """
    Entropy from Information Theory:

     - log_2( (1 /n) * v_counts ) \cdot ( (1 /n) * v_counts )

    """
    if len(list_elements)==0:
        return 0

    iter_counts = construct_dict_count(list_elements).values()
    n = sum(iter_counts)

    array_terms = numpy.array(list(iter_counts))
    scalar = numpy.divide(1, n)
    array_terms_scaled = numpy.multiply(scalar, array_terms)

    return -numpy.dot(numpy.log2(array_terms_scaled), array_terms_scaled)

def calculate_weighted_auxillary(list_elements, index_split, function):
    """
    Weighted Auxillary = (1/num_elements) * ( function(LHS)*card(LHS) + function(RHS)*card(RHS) )

    Function can return scalar or vector
    """
    assert 0 <= index_split < len(list_elements)

    LHS = list_elements[:index_split]
    RHS = list_elements[index_split:]

    scalar = numpy.divide(1, len(list_elements))
    LHS_weighted = numpy.dot(function(LHS), len(LHS))
    RHS_weighted = numpy.dot(function(RHS), len(RHS))

    return scalar*(LHS_weighted + RHS_weighted)

def calculate_weighted_something(list_elements, index_split, function):
    """
    No idea = (1/num_elements) * ( function(LHS) \cdot function(LHS) + function(RHS) \cdot function(RHS) )

    Function can return scalar or vector
    """
    assert 0 <= index_split < len(list_elements)

    LHS = list_elements[:index_split]
    RHS = list_elements[index_split:]

    scalar = numpy.divide(1, len(list_elements))
    LHS_weighted = numpy.dot(function(LHS), function(LHS))
    RHS_weighted = numpy.dot(function(RHS), function(RHS))

    return scalar*(LHS_weighted + RHS_weighted)