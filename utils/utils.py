import re

def read_from_file(filename):
    with open(filename) as f:
        fmt = f.readlines()
        fmt = [x.strip().split(',') for x in fmt]
    return fmt

def read_constraints(filename):
    m = lambda x : re.search(';', x)
    c_info = read_from_file(filename)
    c_info = [[tuple(x.split(';')) if m(x) else x for x in y] for y in c_info]
    return c_info

def read_data(filename):
    data = read_from_file(filename)
    m = lambda t, x : re.search(t, x)
    for i,x in enumerate(data):
        for j,y in enumerate(x):
            if j == 1:
                data[i][j] = int(y)
            elif j > 1:
                data[i][j] = tuple(y.split(';'))
    return data
