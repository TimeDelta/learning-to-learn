#!/bin/bash
find computation_graphs/optimizers -iname \*.py | { while read -s file; do python "$file"; done; }
