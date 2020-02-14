import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import os
from python_utils import list_utils, math_utils

def main():
    krr_results_fpath = os.path.join(os.getcwd(), 'krr_results.dat')
    range_dct = {'n':[2,14], 'l':[4,14], 'c':[0,30], 'g':[0,30],'zeta':[1,1],'R^2':[0,1]}
    for i,y_var in enumerate(list(range_dct.keys())):
        for j,x_var in enumerate(list(range_dct.keys())):

            if x_var == y_var:
                continue

            if 'zeta' == y_var or 'zeta' == x_var:
                continue
            
            color_on = 'R^2'
            append_to_title = '_target1_2mpc'

            fp = np.memmap(krr_results_fpath, dtype='float32', mode='c')
            fp_len = len(fp)
            fp.resize((int(fp_len / 6), 6))
            # columns are: n, l, c, g, zeta, R^2
            col_dct = {'n':0, 'l':1, 'c':2, 'g':3, 'zeta':4, 'R^2': 5}
            axis_label_dct = {'n':'number of radial basis functions',
                            'l':'number of angular basis functions',
                            'c':'cutoff radius',
                            'g':'atomic gaussian standard deviation',
                            'zeta':'exponent the kernel is raised to',
                            'R^2':'Test set R^2'}

            #print('fp', fp)

            # Filter for desired range of values of the parameters
            rows_to_delete = []
            for i,row in enumerate(fp):
                for param_symbol in range_dct:
                    if row[col_dct[param_symbol]] < range_dct[param_symbol][0] or row[col_dct[param_symbol]] > range_dct[param_symbol][-1]:
                        rows_to_delete.append(i)
                        break
            fp = list_utils.multi_delete(fp, rows_to_delete, axis=0)

            #print('filtered fp', fp)

            # Round the matrix
            fp = math_utils.round_matrix(fp, 3, leave_int=True)

            
            fp = list_utils.sort_by_col(fp, col_dct['R^2'], reverse=True)

            df = pd.DataFrame(fp, columns=list(col_dct.keys()))
            df.to_csv('krr_results.csv', index=False)
            if i == 0 and j == 0:
                # print out the top few results
                num_results_to_print = 10
                print('top {} results'.format(num_results_to_print))
                df = pd.DataFrame(fp[:num_results_to_print], columns=list(col_dct.keys()))
                print(df)

            plot_title = y_var + '_vs_' + x_var + append_to_title
            y_axis_label = axis_label_dct[y_var]
            x_axis_label = axis_label_dct[x_var]

            f = plt.figure()
            plt.title(plot_title)
            plt.ylabel(y_axis_label)
            plt.xlabel(x_axis_label)
            colors = fp[:,col_dct[color_on]]
            #plt.scatter(fp[:,col_dct[x_var]], fp[:,col_dct[y_var]], c=colors, norm=Normalize(np.min(colors), np.max(colors)), cmap='hot')
            plt.scatter(fp[:,col_dct[x_var]], fp[:,col_dct[y_var]], c=colors)
            f.savefig(plot_title + '.png', bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()