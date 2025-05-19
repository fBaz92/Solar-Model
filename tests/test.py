import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.module import OptimizedPVModule

test_module = OptimizedPVModule(
    name="BP-MSX120",
    Isc=3.87,
    Voc=42.1,
    Impp=3.56,
    Vmpp=33.7,
    Pmpp=120,
    ki=0.065,
    kv=-0.16,
    kp=-0.5,
    ns=72,
)

# Verify the extraction
verification = test_module.verify_extraction()
