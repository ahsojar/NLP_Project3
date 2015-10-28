def training(fname):
  with open(fname) as f:
    content = f.readlines()
    for line in content:
      print line

training('test.txt')

