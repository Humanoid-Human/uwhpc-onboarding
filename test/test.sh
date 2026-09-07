#!/bin/bash

[ -d "test" ] && cd "test"
python3 "test.py" > "test.hpp"
g++ -O3 -Wall "test.cc" -o "test.out"
./test.out
rm "./test.out"
