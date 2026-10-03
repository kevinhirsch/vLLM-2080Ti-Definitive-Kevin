from compare2 import run
for N in (200,285,400):
  for nf in (0.25,0.5):
    a=run(N,0,0,n_req=1500,noise_frac=nf); b=run(N,0,0,n_req=1500,noise_frac=nf,noise_no_store=True)
    print(f"N={N} noise={nf}: LRU {a[0]:.3f}/agent {a[1]:.3f} | noise prompts not stored {b[0]:.3f}/agent {b[1]:.3f}  agent miss {1-a[1]:.3f}->{1-b[1]:.3f}")
