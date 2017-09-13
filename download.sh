#!/bin/bash
mkdir -p raw
cat raw.txt | wget -P raw -i -
