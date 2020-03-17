Companion paper: Adaptive fourth-order phase field analysis using deep energy minimization
https://authors.elsevier.com/a/1aeGGcAT74SDg

Pre-requisite packages:
Tensorflow 1.15 (Deep Neural Network framework)
geomdl 5.2.9 (Geometry manipulation and plotting)
pyevtk 1.1.1 (VTK output for 3D examples)


For the examples to study fracture analysis the files that have '_save' in the file name can be used to generate the adaptive integration points. The points sets are also provided as ".mat" files. For the actual simulation, the main files are the ones with '_load' in the file name, which load the quadrature points to ensure consistent results. A GPU is recommended, particularly for the 3D and crack propagation examples.


