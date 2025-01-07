import time
import pandas as pd
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--time", type=float, default=1.0, help="time in hours")
    args = parser.parse_args()
    
    for idx in range(int(args.chunks)):
        print(f"Process {idx}")
        if idx == int(args.chunks) - 1:
            subprocess.run([f"python3 dmg_ens_infer.py --time={args.time} --chunk_id={idx} --chunks={args.chunks} "], shell=True)
        else:
            subprocess.Popen([f"python3 dmg_ens_infer.py --time={args.time} --chunk_id={idx} --chunks={args.chunks} "], shell=True)
        time.sleep(1e-2)
#        if idx == int(args.chunks) - 1:
#            subprocess.run([f"python3 update_test.py --chunk_id={idx} --chunks={args.chunks} "], shell=True)
#        else:
#            subprocess.Popen([f"python3 update_test.py --chunk_id={idx} --chunks={args.chunks} "], shell=True)
#        time.sleep(1e-2)
    
    #csv_lst = []
    #for idx in range(int(args.chunks)):
    #    df = pd.read_csv(f'result_test_{idx}.csv')
    #    csv_lst.append(df)
    #pd.concat(csv_lst).to_csv('result_test.csv')
