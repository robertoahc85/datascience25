import numpy as np
tensor_3d = np.arange(1, 13).reshape(2, 3, 4)  # Crea un tensor 3D de forma (2, 3, 4) arreglo multidimesional
print("Tensor 3D:\n", tensor_3d)
print("Dimensiones del tensor 3D:", tensor_3d.ndim)
print("Forma del tensor 3D:", tensor_3d.shape)
print("Tamaño del tensor 3D:", tensor_3d.size)
print("Tipo de datos del tensor 3D:", tensor_3d.dtype)