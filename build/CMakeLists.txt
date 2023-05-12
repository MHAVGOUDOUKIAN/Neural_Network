cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME $ENV{TITLE})
project(${PROJECT_NAME})

# Very important, otherwise your executable will be considered as a shared library by your OS and not as an executable
set (CMAKE_CXX_FLAGS "-no-pie")
set (SRC_DIR ${PROJECT_SOURCE_DIR}/../src)

set(PERMISSIONS OWNER_READ 
    # Set the permissions you want for your executable
    # You can choose between EXECUTE/READ/WRITE for OWNER/GROUP/WORLD
    OWNER_EXECUTE
    OWNER_READ
    OWNER_WRITE
    GROUP_EXECUTE 
    WORLD_EXECUTE
)

include_directories(${SRC_DIR})

# Select the sources and headers files to use in the current directory
set( SOURCES
    ${SRC_DIR}/main.cpp

    #--Engine
    ${SRC_DIR}/Engine/Engine.cpp
    ${SRC_DIR}/Engine/EventHandler.cpp
    
    #--Features
    ${SRC_DIR}/Features/Network/Network.cpp
    
    ${SRC_DIR}/Features/TimeManager/TimeManager.cpp
    
    ${SRC_DIR}/Features/Particles/Particule.cpp
    ${SRC_DIR}/Features/Particles/ParticuleGenerator.cpp
    ${SRC_DIR}/Features/Particles/Ball.cpp

    #--Application
    ${SRC_DIR}/Application/App.cpp
    ${SRC_DIR}/Application/Matrix.cpp
)

set( HEADERS
    #--Engine
    ${SRC_DIR}/Engine/Engine.hpp
    ${SRC_DIR}/Engine/EventHandler.hpp
    
    #--Features
    ${SRC_DIR}/Features/Network/Network.hpp
    
    ${SRC_DIR}/Features/TimeManager/TimeManager.hpp
    
    ${SRC_DIR}/Features/Particles/Particule.hpp
    ${SRC_DIR}/Features/Particles/ParticuleGenerator.hpp
    ${SRC_DIR}/Features/Particles/Ball.hpp
    
    ${SRC_DIR}/Features/Random/Random.hpp
    
    #--Application
    ${SRC_DIR}/Application/App.hpp
    ${SRC_DIR}/Application/Matrix.h
)

add_executable(${PROJECT_NAME} # Name of the executable
    ${SOURCES}
    ${HEADERS})
    
target_link_libraries(${PROJECT_NAME}
    sfml-graphics
    sfml-window
    sfml-system 
    sfml-audio
    sfml-network
    pthread
)

install(FILES ${PROJECT_NAME}  DESTINATION ${PROJECT_SOURCE_DIR}/.. PERMISSIONS ${PERMISSIONS} )
