#!/bin/bash
# Permanent fix for MoviePy regex warnings
find /usr/local/lib/python3.*/site-packages/moviepy/ -type f -name "*.py" -exec sed -i 's/ re.search('\''\\d/ re.search(r'\''\\d/g' {} +
find /usr/local/lib/python3.*/site-packages/moviepy/ -type f -name "*.py" -exec sed -i "s/ re.search('\\d/ re.search(r'\\d/g" {} +
find /usr/local/lib/python3.*/site-packages/moviepy/ -type f -name "*.py" -exec sed -i 's/ re.search('\''\\P/ re.search(r'\''\\P/g' {} +
