import subprocess

# subprocess.call(["python3", "preprocessing.py", "1", "&"])
# subprocess.call(["python3", "preprocessing.py", "2", "&"])
# subprocess.call(["python3", "preprocessing.py", "3", "&"])
# subprocess.call(["python3", "preprocessing.py", "4", "&"])

child_processes = []
for i in range(1, 5):
    p = subprocess.Popen(["python3", "preprocessing.py", str(i)])
    child_processes.append(p)    # start this one, and immediately return to start another

for cp in child_processes:
    cp.wait()
