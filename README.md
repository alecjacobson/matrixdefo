

    git clone --recursive https://github.com/alecjacobson/matrixdefo.git

# build

    mkdir build
    cd build
    cmake ../
    make


# before you run

    cat ../data/face-deltas.dmat.* > ../data/face-deltas.dmat 

# run

    ./matrixdefo ../data/face{.ply,-deltas.dmat}
