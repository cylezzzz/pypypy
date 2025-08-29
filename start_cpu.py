import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Forciere CPU-only
exec(open("start.py").read())