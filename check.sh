#!/bin/bash

echo 
echo checking J1
echo ----------------------------------------------------
echo 

./bin/diff_j1   | tee    check.out

echo 
echo checking J2
echo ----------------------------------------------------
echo 

./bin/diff_j2   | tee -a check.out

echo 
echo checking JeeI
echo ----------------------------------------------------
echo 

./bin/diff_jeeI | tee -a check.out

exit `grep Fail check.out | wc -l`
