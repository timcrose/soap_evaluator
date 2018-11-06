import sys
sys.path.append('/home/trose/python_utils')
import file_utils

def main():
    reranking_files = file_utils.glob('average*reranking*.csv')
    best_rsqrd = 0
    for reranking_file in reranking_files:
        red_csv = file_utils.read_csv(reranking_file)
        rsqrds = [float(row[1]) for row in red_csv]
        if max(rsqrds) > best_rsqrd:
            best_rsqrd = max(rsqrds)
            best_params = reranking_file
    print(best_params, best_rsqrd)

main()
