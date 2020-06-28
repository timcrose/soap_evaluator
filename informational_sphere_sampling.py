import numpy as np
from time import time


def binary_search(labels, target, low, high, selected_indices):
    
    if len(labels) == 0:
        raise Exception('cant search an empty list')
    if high == low:
        visible_set = set(labels[high + 1:]) - set(labels[:high + 1]) - set(selected_indices)
        return high, visible_set
    mid = low + int((high - low) / 2)
    
    visible_set = set(labels[mid + 1:]) - set(labels[:mid + 1]) - set(selected_indices)
    number_visible = len(visible_set)
    
    if number_visible == target:
        return mid, visible_set
    if number_visible < target:
        return binary_search(labels, target, low, mid - 1, selected_indices)
    else:
        return binary_search(labels, target, mid + 1, high, selected_indices)
        
        
def get_r(selected_indices, dist_matrix, n, r, random_if_20pct_remaining=False):
    '''
    selected_indices: np.array shape 1D
        current selected indices of dist_matrix
        
    dist_matrix: np.array shape m x m where m >= n
        pairwise distances of points
        
    n: int
        number of points to select
        
    r: float
        starting r. It should be one of the elements in dist_matrix
        
    return: float, list
        radius on each point that enables num_pts_still_need_to_select to be
        not enveloped in any circle. The list returned is the new selected_indices
        after one more point was selected.
        
    Purpose: Binary search on the distances to find a radius that enables
        num_pts_still_need_to_select to be selectable (i.e. not enveloped in
        any circle).
    '''

    dists_from_selected = dist_matrix[selected_indices]
    dists_from_selected = dists_from_selected.reshape(dists_from_selected.size, 1)
    all_idx = np.repeat(np.array([np.arange(len(dist_matrix))]), len(selected_indices), axis=0).reshape(dists_from_selected.size, 1)
    b = np.concatenate([all_idx, dists_from_selected], axis=1)
    sorted_dists_and_labels = b[b[:, 1].argsort()]
    #sorted_dists_and_labels has its 0th column being the indices in dist_matrix and the 1th column
    # is the corresponding distance to one of the points in selected_indices.
    labels = sorted_dists_and_labels[:, 0]
    sorted_dists= sorted_dists_and_labels[:, 1]
    
    position_of_r = np.where(sorted_dists == r)[0][0]
    visible_set = set(labels[position_of_r + 1:]) - set(labels[:position_of_r + 1]) - set(selected_indices)
    number_visible = len(visible_set)
    target = n - len(selected_indices)
    
    if number_visible < target:
        low = 0
        high = position_of_r
        position_of_r, visible_set = binary_search(labels, target, low, high, selected_indices)
    if len(visible_set) == 0:
        print('empty visible set. number_visible = ' + str(number_visible) + '. target = ' + str(target))
    
    visible_list = list(visible_set)

    if random_if_20pct_remaining:
        if float(len(selected_indices)) / float(len(dist_matrix)) >= 0.8:
            selected_indices.append(int(np.random.choice(np.array(visible_list), 1)[0]))
            return sorted_dists[position_of_r], selected_indices


    max_total_dist = -1
    for pt in visible_list:
        pt = int(pt)
        total_dist = dist_matrix[selected_indices, [pt]].sum()
        if total_dist > max_total_dist:
            max_total_dist = total_dist
            candidate = pt
    try:
        selected_indices.append(candidate)
    except:
        print('dist_matrix', flush=True)
        print(dist_matrix, flush=True)
        print('number_visible', flush=True)
        print(number_visible, flush=True)
        print('target', flush=True)
        print(target, flush=True)
        print('total_dist', flush=True)
        print(total_dist, flush=True)
        print('max_total_dist', flush=True)
        print(max_total_dist, flush=True)
        raise Exception('should have gotten an candidate')
        
    return sorted_dists[position_of_r], selected_indices


def influential_sphere_sampling(similarity_matrix, n_to_select, random_if_20pct_remaining=False):

    sim_mat_shape = similarity_matrix.shape
    similarity_diagonal = np.diag(similarity_matrix)
    sim_to_dist_start_time = time()
    dist_matrix = np.repeat(similarity_diagonal, sim_mat_shape[0]).reshape(sim_mat_shape) + similarity_diagonal - 2 * similarity_matrix
    print('time sim to dist', time() - sim_to_dist_start_time, flush=True)
    if np.min(dist_matrix) < -1e-3:
        print('might have a problem with the kernel', np.min(dist_matrix), flush=True)
    dist_matrix[np.where(dist_matrix < 0)] = 0.0
    #print('dist_matrix', dist_matrix, flush=True)
    # Only doing the following horrendous version of np.sqrt(dist_matrix) because I'm getting a weird "divide by zero in sqrt" message
    for i,row in enumerate(dist_matrix):
        for j,entry in enumerate(row):
            #print('np.sqrt(entry)', np.sqrt(entry), flush=True)
            dist_matrix[i][j] = np.sqrt(entry)
    print('got dist_matrix', flush=True)
    selected_indices = [np.random.randint(0, len(dist_matrix))]
    r = dist_matrix[selected_indices[0]].max()
    if n_to_select > len(dist_matrix):
        raise Exception('You requested n = ' + str(n_to_select) + ' points but only ' +
                        'provided ' + str(len(dist_matrix)) + ' points.')
    elif n_to_select <= 0:
        raise Exception('You requested n = ' + str(n_to_select) + ' points but only n > 0' + 
                            ' is valid')
    
    num_selected = 1
    start_time_for_pts = time()
    print('starting selection', flush=True)
    while num_selected < n_to_select:
        time_for_one_pt = time()
        r, selected_indices = get_r(selected_indices, dist_matrix, n_to_select, r, random_if_20pct_remaining=random_if_20pct_remaining)
        print('time for one point', time() - time_for_one_pt, flush=True)
        num_selected += 1
    print('time for', n_to_select, 'points', time() - start_time_for_pts, flush=True)
    return np.array(selected_indices), r


def get_centered_L2_discrepancy(data, selected_indices):
    import openturns as ot
    
    selected_indices = list(map(int,list(selected_indices)))
    design = data[selected_indices]
    crit = ot.SpaceFillingC2().evaluate(design)
    return crit


def plot_data(data, selected_indices, r, type_of_sampling_method):
    import matplotlib
    import matplotlib.pyplot as plt
    
    c = np.array(['black'] * len(data))
    selected_indices = list(map(int,list(selected_indices)))
    c[selected_indices] = 'red'
    fig, ax = plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=c, s=100)
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 16
    fig_size[1] = 16
    plt.rcParams['figure.figsize'] = fig_size
    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)
    
    fontdict = {'fontsize': 32,
                'fontweight' : plt.rcParams['axes.titleweight'],
                'verticalalignment': 'bottom',
                'horizontalalignment': 'center'}
    
    plt.title(type_of_sampling_method + ' sampling demo', fontdict=fontdict)
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    for selected_idx in selected_indices:
        xy = tuple(data[selected_idx])
        circle = plt.Circle(xy, r, color='r', alpha=0.25)
        ax.add_artist(circle)
        
    plt.show()


def test():
    import sklearn.metrics
    
    times = []
    data = np.random.rand(50,2)
    n_to_select = 20
    dist_matrix = sklearn.metrics.pairwise_distances(data)
    similarity_matrix = np.max(dist_matrix) - dist_matrix
    for i in range(10):
        start_time = time()
        selected_indices, r = influential_sphere_sampling(similarity_matrix, n_to_select)
        times.append(time() - start_time)
    print(type(selected_indices))
    plot_data(data, selected_indices, r, 'influential_sphere_sampling')
    print(get_centered_L2_discrepancy(data, selected_indices))
    print(sum(times) / float(len(times)))
    
    
if __name__ == '__main__':
    test()
