#!/bin/bash 
cp rl.cofig.0 rl.config
python train.py

cp rl.cofig.1 rl.config
python train.py
