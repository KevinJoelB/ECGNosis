import h5py


file = h5py.File("exams_part17.hdf5", "r")


def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f" Group: {name}")

file.visititems(print_structure)
file.close()