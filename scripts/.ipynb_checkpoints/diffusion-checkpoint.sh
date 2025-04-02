#!/bin/bash
#
# Simulate diffusion
# Replace all paths here manually

function usage {
    echo "usage: $0 -i infile -o outfolder [-r outprefix] [-p parameter_file] [-n pnumber]"
    echo "  -i infile           specify name of the input params file"
    echo "  -o outfolder        specify name the first prefix of the output folder"
    echo "  -r outprefix        specify name the second prefix of the output folder"
    echo "  -p parameter_file   specify name of the parameter file"
    echo "  -n pnumber          specify parameter number from the parameter file (indexed from 1)"
    exit 1
}

pos_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i ) i="$2"; shift 2;;
    -o ) o="$2"; shift 2;;
    -r ) r="$2"; shift 2;;
    -p ) p="$2"; shift 2;;
    -n ) n="$2"; shift 2;;
    *)
  esac
done

path='/home/artem.kays/scripts/lncRNAeffects/RNA_diffusion'


start=$(date +%s)
mkdir -p $path/results/$o
printf "Start: $(date)\n" >> $path/results/$o/log.txt
# printf "$r\n" >> $path/results/$o/log.txt

# nice -n 19
{ time nice -n 18 python3 $path/scripts/rna_diffusion.py --i $i --o $o --r ${r:-""} --p ${p:-""} --pN ${n:-""} ; } \
    &> >(grep -v -e '^[[:space:]]*$' -e ALSA -e PYGAME -e Moviepy -e ▊ &>> $path/results/$o/log.txt)
    
end=$(date +%s)
# printf "Duration: $(date -d@$(($end-$start)) -u +%H:%M:%S)\n" >> $path/results/$o/log.txt
printf "Finish: $(date)\n\n\n\n" >> $path/results/$o/log.txt