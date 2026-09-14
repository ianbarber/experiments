# Mutation operators (applied to a near-boundary task the current policy passes 1-3 times out of 8)

1. add-constraint: keep the objective, add one verifiable constraint (format, naming, edge case, size limit).
2. change-format: same data, different output format or schema (CSV -> JSON, flat -> nested, one file -> two).
3. compose-skill: require one more skill domain (add a shell step to a python task, a sqlite query to a file task, a test to a debugging task).
4. harden-inputs: introduce a realistic wrinkle in the inputs (missing values, encoding, duplicates, timezone, trailing whitespace, mixed line endings).
5. invert: given the outputs, produce or repair the inputs/config that generate them.
6. scale: more files, more records, deeper directory tree; solution must generalise (no hard-coded lists).
7. debug-variant: plant a different bug of the same class in the same code.
