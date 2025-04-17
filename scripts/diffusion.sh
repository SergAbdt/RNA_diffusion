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

path='/home/artem.kays/scripts/RNA_diffusion'
log="$path/results/$o/log.txt"

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>>$log 2>&1


start=$(date +%s)
echo "Start: $(date)\n" >&3
echo "Start: $(date)\n"
mkdir -p $path/results/$o
# printf "$r\n"

{ time nice -n 19 python3 $path/scripts/rna_diffusion.py --i $i --o $o --r ${r:-""} --p ${p:-""} --pN ${n:-""} ; } \
    &> >(grep -v -e '^[[:space:]]*$' -e ALSA -e PYGAME -e Moviepy -e ▊)
    
end=$(date +%s)
echo "Duration: $(date -d@$(($end-$start)) -u +%H:%M:%S)\n" >&3
echo "Finish: $(date)"
printf "\n\n\n"
echo "$(date) : Done" >&3