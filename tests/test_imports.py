import importlib.metadata
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import seaborn as sns
import streamlit as st
import plotly
import jupyter_core

def test_imports():
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert scipy.__version__ is not None
    assert sklearn.__version__ is not None
    assert matplotlib.__version__ is not None
    assert sns.__version__ is not None
    assert st.__version__ is not None
    assert plotly.__version__ is not None
    assert jupyter_core.__version__ is not None

if __name__ == '__main__':
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"seaborn: {sns.__version__}")
    print(f"streamlit: {st.__version__}")
    print(f"plotly: {plotly.__version__}")
    print(f"jupyter: {importlib.metadata.version('jupyter')}")
    print(f"jupyter_core: {jupyter_core.__version__}")
    print("ALL PACKAGES IMPORTED SUCCESSFULLY!")
