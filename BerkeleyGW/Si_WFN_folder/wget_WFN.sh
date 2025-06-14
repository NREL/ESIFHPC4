#!/bin/bash

Usage () {
  echo "wget_WFN.sh: a script for retrieving WFN files for the ESIF-HPC4 BerkeleyGW benchmark."
  echo "Usage: wget_WFN.sh <size>"
  echo "Allowed sizes:"
  echo "[ small     (   3 GB )," 
  echo "  medium    (  18 GB ),"
  echo "  large     (  71 GB ) ]"
}

#ESIF_HPC4=<TODO>
#Si_WFN_url=<TODO>
NERSC_TEN=https://portal.nersc.gov/project/m888/nersc10
Si_WFN_url=$NERSC_TEN/benchmark_data/BGW_input


case "$1" in

  --help)
  Usage
  exit
  ;;

  small)
    WFN_gz=Si214_WFN_file.tar.gz
    ;;

  medium)
    WFN_gz=Si510_WFN_file.tar.gz
    ;;

  large)
    WFN_gz=Si998_WFN_file.tar.gz
    ;;

  *)
    echo "Error: the requested WFN size ($1) is not supported.)"
    Usage
    exit 1

esac

wget $Si_WFN_url/$WFN_gz
tar -xvf $WFN_gz

