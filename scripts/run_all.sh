#!/usr/bin/env bash
python vision/baseline.py
python vision/prune.py
python vision/quantize.py
python nlp/baseline.py
python nlp/prune.py
python nlp/quantize.py