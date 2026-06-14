# Legacy Code

**`controllers_cvxpy_legacy.py`** is a prototype implementation of SMPC and DMPC that uses
CVXPY as the modeling layer. It was used in early development but is not the version that
produced the paper's results.

The production controllers used for all published results are in `../controllers_improved.py`,
which calls OSQP directly (via its Python interface) for significantly lower per-step latency.

If you need CVXPY support (e.g., for rapid prototyping or adding new constraints via the
CVXPY DSL), this file provides a working starting point. It requires `cvxpy>=1.4.0` in
addition to the packages in `requirements.txt`.
