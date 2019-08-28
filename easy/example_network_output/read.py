import h5py
filename = 'synapse_positions.h5'
f = h5py.File(filename, 'r')

a = f.items()

list(a)

print(f["E:E"][()])

f.close()