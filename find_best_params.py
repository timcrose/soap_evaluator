import sys
sys.path.append(os.path.join(os.environ["HOME"], "python_utils")))
import file_utils

def main():
    pwd = file_utils.os.getcwd()
    reranking_files = file_utils.glob(file_utils.os.path.join(pwd, 'average*reranking*.csv'))
    best_rsqrd = 0
    for reranking_file in reranking_files:
        red_csv = file_utils.read_csv(reranking_file)
        rsqrds = [float(row[1]) for row in red_csv]
        if max(rsqrds) > best_rsqrd:
            best_rsqrd = max(rsqrds)
            best_params = reranking_file
    print(best_params, best_rsqrd)

main()
