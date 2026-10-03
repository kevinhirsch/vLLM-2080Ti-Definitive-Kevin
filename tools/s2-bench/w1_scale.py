import subprocess,sys,json,os
tag=sys.argv[1]; env=dict(os.environ)
for n in (1,4,8):
    subprocess.run(f"python3 estate_load.py --n {n} --min-tok 18000 --max-tok 36000 --max-tokens 1 >/dev/null 2>&1",shell=True,env=env)
    r=subprocess.run(f"python3 estate_load.py --n {n} --min-tok 18000 --max-tok 36000 --max-tokens 200 --out scale_{tag}_{n}.json",shell=True,env=env,capture_output=True,text=True)
    print(tag,"ctx25k N=%d"%n,r.stdout.strip()[-200:])
for n in (1,12):
    r=subprocess.run(f"python3 conc_short.py {n} 300",shell=True,capture_output=True,text=True); print(tag,"short",r.stdout.strip())
