import sys, getopt
import pandas as pd
import json
import matplotlib.pyplot as plt


#datasets = "../Datasets/"
#df = pd.read_json(datasets+argv[1])

#df["ultimoPrecio"]

#dt=df[['fechaHora','ultimoPrecio',]]
#dt=dt.set_index(pd.DatetimeIndex(dt['fechaHora']))

#dt=dt.sort_index(ascending=True)
#dt['ultimoPrecio'].plot()
#plt.show()



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
   df = pd.read_json(inputfile)
   dt=df[['fechaHora','ultimoPrecio',]]
   dt=dt.set_index(pd.DatetimeIndex(dt['fechaHora']))
   dt.to_csv(str(outputfile))
   print('Output file is "', outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])
