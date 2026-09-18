#!/bin/bash

[ -d "test" ] && cd "test"
python3 "test.py" > "test.hpp"
g++ -O3 -g "test.cc" -o "test.out"
./test.out
rm "./test.out"
