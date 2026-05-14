agents

- language: Julia
- core docs: https://docs.julialang.org/
- differential equations: https://diffeq.sciml.ai/stable/
- stochastic (SDE) docs: https://diffeq.sciml.ai/stable/tutorials/stochastic_differential_equations/ and https://diffeq.sciml.ai/stable/solvers/sde_solve/
- packages: DifferentialEquations.jl (includes StochasticDiffEq solvers)

style

- prefer simple, idiomatic Julia.
- do NOT add defensive programming unless strictly necessary.
- keep scripts concise and direct; simulation scripts should be single-file runnable scripts with no helper functions.

notes

- add Pkg.add("DifferentialEquations") when needed; use `using DifferentialEquations` in scripts.
- keep documentation links above for quick reference.
