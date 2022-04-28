


    git clone --recursive https://github.com/alecjacobson/matrixdefo.git

# build

    mkdir build
    cd build
    cmake ../
    make

# run elephant

    ./matrixdefo ../data/elephant{.obj,-modes.dmat}
    


https://user-images.githubusercontent.com/2241689/165771575-ddb12b10-5f95-4ba7-875a-19705f160262.mp4



# before you run faces

    cat ../data/face-deltas.dmat.* > ../data/face-deltas.dmat 

# run faces

    ./matrixdefo ../data/face{.ply,-deltas.dmat}
    
    
_23,728 vertices deformed using 384 principle component vectors in the vertex shader by packing the 23,728×384 matrix in a 3019² texture._

https://user-images.githubusercontent.com/2241689/165660420-38aade02-31d2-4186-9cda-b176ab569e29.mp4
