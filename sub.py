import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

start = time.perf_counter()

for idx in range(10):
    if idx == int(10) - 1:
        subprocess.run(["python3 vroom.py"], shell=True)
    else:
        subprocess.Popen(["python3 vroom.py"], shell=True)

print(time.perf_counter() - start)

