#!/bin/sh

# Change this env variable to change the name of the executable generated with cmake
export TITLE="SFML_App"

printb() {
    tput setaf $2
    tput bold
    echo $1
    tput sgr0
}

printb "#==== Launching installation ====#" 2
cd build
# Set the argument "DCMAKE_BUILD_TYPE" to Debug or Release
cmake . -DCMAKE_BUILD_TYPE=Release
cmake --build . 
cmake --install .
rm $TITLE
cd ..

printb "#==== Installation finished ====#" 2
