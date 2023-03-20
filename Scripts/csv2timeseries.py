import sys, getopt
import pandas as pd
import json
import matplotlib.pyplot as plt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('json2csv.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('json2csv.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = str(argv[1])
      elif opt in ("-o", "--ofile"):
         outputfile = str(argv[3])
   print('Input file is "', inputfile)
   df = pd.read_csv(inputfile)
   df=df[['fechaHora','ultimoPrecio',]]
   df=df.set_index(pd.DatetimeIndex(df['fechaHora']))
   df.plot()
   plt.title("timeseries: "+inputfile)
   plt.xlabel("time")
   plt.ylabel("AR$ ")
   plt.grid()
   plt.savefig(outputfile)
   print('Output file is "', outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])


