#!/usr/bin/env python3

import argparse, re

def main(args):
  files = args['files']
  if len(files) == 0: files = ["rsl.out.0000"]
  max_fname = max([len(f) for f in files]+[4])

  hdr_row1="File | Comp Steps |     Comp Total |       Comp Avg |     Init Total |       Init Avg |    Write Total |      Write Avg |  X  |  Y  |"
  hdr_row2="-----+------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+-----|"
  row_fmt = "| {:"+str(max_fname)+"} | {:10d} | {:14.6f} | {:14.6f} | {:14.6f} | {:14.6f} | {:14.6f} | {:14.6f} | {:3d} | {:3d} |"
  print("| ", " "*(max_fname-4), hdr_row1, sep="")
  print("|-", "-"*(max_fname-4), hdr_row2, sep="")

  cpu_regex = r'Ntasks in X\s*(\d+)\s*,\s*ntasks in Y\s*(\d+)'
  main_regex = r'\s*Timing for main: time [0-9_:-]+ on domain'
  write_regex = r'Timing for Writing [a-zA-Z0-9_:-]+ for domain'
  secs_regex = r'\s*\d+:\s*(\d+\.\d+)\s*elapsed seconds'

  for fname in files:
    with open(fname, 'r') as f:
      txt = f.read()
      cpu_match = re.search(cpu_regex, txt)
      X, Y = int(cpu_match.group(1)), int(cpu_match.group(2))

      write = [float(t) for t in re.findall(write_regex+secs_regex, txt)]

      init = []; comp = []
      for m in re.finditer(r'('+main_regex+secs_regex+r')+', txt):
        secs = [float(t) for t in re.findall(main_regex+secs_regex, m.group())]
        init += [secs[0]]
        comp += secs[1:]

      print(row_fmt.format(
        fname, len(comp), sum(comp), sum(comp)/len(comp), sum(init), sum
        (init)/len(init), sum(write), sum(write)/len(write), X, Y))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Analyzes timing output info for WRF runs. If no rsl output "
    "files are specified then look for rsl.out.0000 in current directory")
  parser.add_argument("files", nargs="*", help= "rsl.out.0000 file")
  main(vars(parser.parse_args()))
