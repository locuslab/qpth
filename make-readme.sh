#!/bin/bash

python2 -m readme2tex --output README.md --readme README-IN.md \
        --svgdir 'svgs'
sed -i 's|https://rawgit.com/in\tgit@github.com:locuslab/qpth/None|https://rawgit.com/locuslab/qpth/master|g' README.md
