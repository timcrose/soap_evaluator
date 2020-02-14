from math import factorial
from random import randint
from warnings import filterwarnings
filterwarnings('ignore', message='The objective has been evaluated at this point before.')
filterwarnings('ignore', category=FutureWarning)
from skopt import Optimizer
from skopt.space import Real, Integer

def func(a,b):
    return a**2 + b

def get_search_space_size(Int_vars):
    '''
    Int_vars: list of list of float or int
        These are the variables that are in your
        skopt.Optimizer object. They each have a list
        of length 2 which defines the
        upper and lower bound this variable take on.

    Return: int
        number of combinations of the ranges of Int_vars.
        e.g. Int_vars = [Integer(1, 3), Integer(3,4)]
        then because there are 6 combinations:
        [(1,3),(1,4),(2,3),(2,4),(3,3),(3,4)]
        then return 6. This is gotten by (3-1+1) * (4-3+1)

    Purpose: In order to make sure you dont ask in an infinite
        loop for more points when youve searched them all, you
        need to know the number of total possible combinations.
    '''
    num_combinations = 1
    for Int_var in Int_vars:
        num_combinations *= Int_var[1] - Int_var[0] + 1
    return num_combinations

def tell_opt(opt, pts, output_values):
    '''
    opt: Optimizer
        Optimizer object which should have had a dct dictionary
        attribute created for it prior to using this function. The
        dct attribute should have a key being a tuple of parameters
        and the value being the output value of the function one is
        trying to optimize.

    pts: list of list of numbers
        pts are the points that were evaluated on some function that
        you are trying to optimize and that gave the output function
        values given in output_values in corresponding order.
    
    output_values: list of numbers
        List of output functions values when pts are evaluated on a
        function you are trying to optimize. The order should correspond
        the order of the pts.

    Return: Optimizer
        The Optimizer object now has its dictionary of output input pairs
        updated. I store these in a dictionary for convenience in searching
        for the output value of a particular input parameter set as opposed to
        manipulating and searching the opt.Xi and opt.yi attributes. This is acceptable
        unless this dictionary takes too much memory.

    Purpose: Update an Optimizer's input / output point pair values in a dictionary which should
        already have been made by the user prior to running this function. Also, tell the Optimizer
        the pts and output_values.
    '''
    opt.tell(pts, output_values)
    for i,pt in enumerate(pts):
        opt.dct[tuple(pt)] = output_values[i]
    return opt

def ask_opt(opt, Int_vars, search_space_size, num_novel_pts_to_get=1, max_iterations=100, strategy='cl_min'):
    '''
    opt: Optimizer
        Optimizer object which should have had a dct dictionary
        attribute created for it prior to using this function. The
        dct attribute should have a key being a tuple of parameters
        and the value being the output value of the function one is
        trying to optimize.

    Int_vars: list of skopt.space.Integer
        These are the variables that are in your
        skopt.Optimizer object.

    search_space_size: int
        Number of possible parameter combinations given that all parameters are restricted to integers.

    num_novel_pts_to_get: int
        Number of input points to get to later be used on an objective function you
        are trying to optimize. Novel means points that have not already been evaluated.

    max_iterations: int
        Number of attempts to get a novel input parameter set from opt before giving up and
        filling in remaining spots with random novel input parameter sets. This is needed because opt.ask
        will possibly return points that you've already evaluated. If your objective function is noisy, then
        it is fine to evaluate these parameter sets again, but this function is only used when your function
        is not noisy enough to want to evaluate the same parameter sets again.

    strategy: str
        See Optimizer.ask documentation.

    Return:
    novel_pts: list of list of integers or None
        List of parameter sets (which have length equal to the number of variables) to try next on the objective 
        function. They are either integers or None. They are only None if there are no more novel parameter 
        sets left to try. For example, if novel_pts is [[1,2],[3,4],[None,None],[None,None],[None,None]], then 
        [1,2] and [3,4] are parameter novel sets and the 3 [None,None] represents that there were 5 parameter sets 
        requested but there were only 2 more points in the entire search space.

    Purpose: opt.ask will possibly return points that you've already evaluated. If your objective function is noisy, 
        then it is fine to evaluate these parameter sets again, but this function is only used when your function
        is not noisy enough to want to evaluate the same parameter sets again. It asks opt.ask for new points that
        you haven't evaluated before max_iterations times. If it does not get num_novel_pts_to_get in max_iterations
        iterations, then ask_opt will fill the remaining number of points with random parameter sets that are novel...
        unless there are no more novel points that you haven't already evaluated or included in novel_pts already; in
        this case, it will fill the remaining points with None (see novel_pts Return value description above).
    '''
    num_params = len(Int_vars)
    Xi_tuple = tuple(map(tuple,opt.Xi))
    #print('Xi_tuple',Xi_tuple)
    Xi_set = set(Xi_tuple)
    #print('Xi_set', Xi_set)
    novel_pts = []
    num_random_pts_to_get = 0
    i = 0
    while len(novel_pts) + num_random_pts_to_get < num_novel_pts_to_get and i < max_iterations and len(novel_pts) + len(Xi_set) < search_space_size:
        n_points = min(search_space_size - (len(novel_pts) + len(Xi_set)), num_novel_pts_to_get - (len(novel_pts) + num_random_pts_to_get))
        #print('n_points', n_points)
        pts = opt.ask(n_points=n_points, strategy='cl_min')
        #print('pts', pts)
        pts_tuple = tuple(map(tuple,pts))
        pts_set = set(pts_tuple)
        #print('pts_set', pts_set)
        Xi_tuple = tuple(map(tuple,opt.Xi))
        Xi_set = set(Xi_tuple)
        #print('opt.Xi', opt.Xi)
        #print('Xi_set', Xi_set)
        novel_pts_tuple = tuple(map(tuple, novel_pts))
        novel_pts_set = set(novel_pts_tuple)
        novel_pts_gotten = pts_set - Xi_set - novel_pts_set
        #print('novel_pts_gotten', novel_pts_gotten)
        non_novel_pts_gotten = pts_set.intersection(Xi_set)
        #print('non_novel_pts_gotten', non_novel_pts_gotten)
        for non_novel_pt in non_novel_pts_gotten:
            #print('non_novel_pt', non_novel_pt)
            for _ in range(pts_tuple.count(non_novel_pt)):
                opt.tell(list(non_novel_pt), opt.dct[non_novel_pt])

        for novel_pt in novel_pts_gotten:
            num_random_pts_to_get += pts_tuple.count(novel_pt) - 1
            #print('num_random_pts_to_get', num_random_pts_to_get)
            if len(novel_pts) < num_novel_pts_to_get and novel_pt not in novel_pts:
                #print('appending novel_pt', novel_pt)
                novel_pts.append(list(novel_pt))
        #print('novel_pts', novel_pts)
        #print('len(novel_pts)', len(novel_pts))
        i += 1
    num_random_pts_to_get = min(num_novel_pts_to_get, search_space_size - len(Xi_set)) - len(novel_pts)
    #print('num_random_pts_to_get final', num_random_pts_to_get)
    #print('ask pts', novel_pts)
    if num_random_pts_to_get < 0:
        print('warning, num_random_pts_to_get was found to be < 0, specifically, it is:',num_random_pts_to_get)
    if num_random_pts_to_get > 0:
        #print('len(novel_pts)', len(novel_pts))
        i = 0
        while i < num_random_pts_to_get:
            random_pt = tuple([randint(Int_var[0], Int_var[1]) for Int_var in Int_vars])
            if random_pt not in tuple(map(tuple,novel_pts)) and random_pt not in Xi_set:
                #print('appending random_pt', random_pt)
                novel_pts.append(list(random_pt))
                i += 1
                #print('novel_pts', novel_pts)
                #print('len(novel_pts)', len(novel_pts))
    if num_novel_pts_to_get < len(novel_pts):
        raise Exception('number of novel points should not be greater than num_novel_pts_to_get',
                'num_novel_pts_to_get', num_novel_pts_to_get, 'len(novel_pts)', len(novel_pts))
    novel_pts += [[None] * num_params] * (num_novel_pts_to_get - len(novel_pts))
    return novel_pts

def main():
    a_lower_bound, a_upper_bound = -3, 3
    b_lower_bound, b_upper_bound = -2, 2
    a_Int = Integer(a_lower_bound, a_upper_bound)
    b_Int = Integer(b_lower_bound, b_upper_bound)
    #print('dir(a_Int)', dir(a_Int))
    #print('a_Int.bounds', a_Int.bounds)
    #print('a_Int.low', a_Int.low)
    #print('a_Int.high', a_Int.high)
    Int_vars = [a_Int,b_Int]
    num_params = len(Int_vars)
    search_space_size = get_search_space_size(Int_vars)
    #print('search_space_size', search_space_size)
    
    opt = Optimizer(Int_vars,n_initial_points=2)
    #print('dir(opt)', dir(opt))
    #print('opt.Xi', opt.Xi, 'opt.yi', opt.yi)
    opt.dct = {}
    pts = ask_opt(opt, Int_vars, search_space_size, num_novel_pts_to_get=2, max_iterations=15, strategy='cl_min')
    #pts = opt.ask(n_points=2, strategy='cl_min')
    #print('opt.Xi', opt.Xi, 'opt.yi', opt.yi)
    # pts is a list of list even if n_points=1
    #print('initial pts', pts)
    output_values = [func(a, b) for a,b in pts]
    #print('output_values', output_values)
    opt = tell_opt(opt, pts, output_values)
    #print('opt.Xi', opt.Xi, 'opt.yi', opt.yi)
    #print('opt.dct', opt.dct)
    
    num_novel_pts_to_get = 40
    max_iterations = 15
    novel_pts = ask_opt(opt, Int_vars, search_space_size, num_novel_pts_to_get, max_iterations, strategy='cl_min')

    #print('novel_pts', novel_pts)
    #print('len(novel_pts)', len(novel_pts))
    Xi_tuple = tuple(map(tuple,opt.Xi))
    Xi_set = set(Xi_tuple)
    novel_pts_tuple = tuple(map(tuple, novel_pts))
    novel_pts_set = set(novel_pts_tuple)
    #print('len(novel_pts_set)', len(novel_pts_set))
    novel_pts_gotten = novel_pts_set - Xi_set
    #print('novel_pts_gotten', novel_pts_gotten)
    #print('len(novel_pts_gotten)', len(novel_pts_gotten))
    non_novel_pts_gotten = novel_pts_set.intersection(Xi_set)
    #print('non_novel_pts_gotten', non_novel_pts_gotten)

if __name__ == '__main__':
    main()
