# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o: ../src/IR_SaliencyExtraction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o -c /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/IR_SaliencyExtraction.cpp

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/IR_SaliencyExtraction.cpp > CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.i

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/IR_SaliencyExtraction.cpp -o CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.s

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.requires

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.provides: CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.provides

CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.provides.build: CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o


CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/main.cpp.o -c /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/main.cpp.o.requires

CMakeFiles/main.dir/src/main.cpp.o.provides: CMakeFiles/main.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/main.cpp.o.provides

CMakeFiles/main.dir/src/main.cpp.o.provides.build: CMakeFiles/main.dir/src/main.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o" \
"CMakeFiles/main.dir/src/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

../bin/main: CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o
../bin/main: CMakeFiles/main.dir/src/main.cpp.o
../bin/main: CMakeFiles/main.dir/build.make
../bin/main: /usr/local/lib/libopencv_gapi.so.4.6.0
../bin/main: /usr/local/lib/libopencv_highgui.so.4.6.0
../bin/main: /usr/local/lib/libopencv_ml.so.4.6.0
../bin/main: /usr/local/lib/libopencv_objdetect.so.4.6.0
../bin/main: /usr/local/lib/libopencv_photo.so.4.6.0
../bin/main: /usr/local/lib/libopencv_stitching.so.4.6.0
../bin/main: /usr/local/lib/libopencv_video.so.4.6.0
../bin/main: /usr/local/lib/libopencv_videoio.so.4.6.0
../bin/main: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
../bin/main: /usr/local/lib/libopencv_dnn.so.4.6.0
../bin/main: /usr/local/lib/libopencv_calib3d.so.4.6.0
../bin/main: /usr/local/lib/libopencv_features2d.so.4.6.0
../bin/main: /usr/local/lib/libopencv_flann.so.4.6.0
../bin/main: /usr/local/lib/libopencv_imgproc.so.4.6.0
../bin/main: /usr/local/lib/libopencv_core.so.4.6.0
../bin/main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: ../bin/main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/IR_SaliencyExtraction.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build /home/lx/IR_Enhance_Methods/SaliencyExtractionMethods/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

