"""Pytest root configuration and environment stabilization."""

import os

# Stabilize OpenBLAS/MKL threads on Windows to prevent thread pool exhaustion and C malloc errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
