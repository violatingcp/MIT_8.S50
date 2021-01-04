#!/bin/bash

input=LIGOmanual

filter=" -vE ^$|usr|log|srcs|figures/|hyphenation|information|\[|\]|<|>|restricted|entering|\(|\)|rotated|/c"
[[ $1 == '--debug' ]] && { filter="-v xxxxxx" ; }

echo -e '\e[36;1m===> Pre-compilation\e[0m'
pdflatex -interaction=nonstopmode ${input} &> /dev/null
echo -e '\e[36;1m===> Bibtex-compilation\e[0m'
bibtex ${input} | grep -vE Warning
pdflatex -interaction=nonstopmode ${input} &> /dev/null
echo -e '\e[36;1m===> Final-compilation\e[0m'
pdflatex ${input} | grep $filter
echo -e '\e[36;1m===> Finish-compilation\e[0m'

mkdir -p logs
mv ${input}.aux ${input}.log ${input}.bbl ${input}.blg ${input}.out logs/
mkdir -p logs/srcs
