# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xduser/LiHuan/superpixel_crf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xduser/LiHuan/superpixel_crf

# Include any dependencies generated for this target.
include CMakeFiles/superpixel_crf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/superpixel_crf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/superpixel_crf.dir/flags.make

CMakeFiles/superpixel_crf.dir/j2seg.o: CMakeFiles/superpixel_crf.dir/flags.make
CMakeFiles/superpixel_crf.dir/j2seg.o: j2seg.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/xduser/LiHuan/superpixel_crf/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/superpixel_crf.dir/j2seg.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/superpixel_crf.dir/j2seg.o -c /home/xduser/LiHuan/superpixel_crf/j2seg.cpp

CMakeFiles/superpixel_crf.dir/j2seg.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/superpixel_crf.dir/j2seg.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/xduser/LiHuan/superpixel_crf/j2seg.cpp > CMakeFiles/superpixel_crf.dir/j2seg.i

CMakeFiles/superpixel_crf.dir/j2seg.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/superpixel_crf.dir/j2seg.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/xduser/LiHuan/superpixel_crf/j2seg.cpp -o CMakeFiles/superpixel_crf.dir/j2seg.s

CMakeFiles/superpixel_crf.dir/j2seg.o.requires:
.PHONY : CMakeFiles/superpixel_crf.dir/j2seg.o.requires

CMakeFiles/superpixel_crf.dir/j2seg.o.provides: CMakeFiles/superpixel_crf.dir/j2seg.o.requires
	$(MAKE) -f CMakeFiles/superpixel_crf.dir/build.make CMakeFiles/superpixel_crf.dir/j2seg.o.provides.build
.PHONY : CMakeFiles/superpixel_crf.dir/j2seg.o.provides

CMakeFiles/superpixel_crf.dir/j2seg.o.provides.build: CMakeFiles/superpixel_crf.dir/j2seg.o

# Object files for target superpixel_crf
superpixel_crf_OBJECTS = \
"CMakeFiles/superpixel_crf.dir/j2seg.o"

# External object files for target superpixel_crf
superpixel_crf_EXTERNAL_OBJECTS =

superpixel_crf: CMakeFiles/superpixel_crf.dir/j2seg.o
superpixel_crf: lib/libSLIC.a
superpixel_crf: lib/libopencv_lbp.a
superpixel_crf: /usr/local/lib/libopencv_videostab.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_video.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_ts.a
superpixel_crf: /usr/local/lib/libopencv_superres.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_stitching.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_photo.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_objdetect.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_nonfree.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_ml.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_legacy.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_imgproc.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_highgui.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_gpu.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_flann.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_features2d.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_core.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_contrib.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_calib3d.so.2.4.10
superpixel_crf: lib/libcnpy.a
superpixel_crf: lib/libdensecrf.a
superpixel_crf: /usr/local/lib/libopencv_nonfree.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_gpu.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_photo.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_objdetect.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_legacy.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_video.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_ml.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_calib3d.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_features2d.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_highgui.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_imgproc.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_flann.so.2.4.10
superpixel_crf: /usr/local/lib/libopencv_core.so.2.4.10
superpixel_crf: CMakeFiles/superpixel_crf.dir/build.make
superpixel_crf: CMakeFiles/superpixel_crf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable superpixel_crf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/superpixel_crf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/superpixel_crf.dir/build: superpixel_crf
.PHONY : CMakeFiles/superpixel_crf.dir/build

CMakeFiles/superpixel_crf.dir/requires: CMakeFiles/superpixel_crf.dir/j2seg.o.requires
.PHONY : CMakeFiles/superpixel_crf.dir/requires

CMakeFiles/superpixel_crf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/superpixel_crf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/superpixel_crf.dir/clean

CMakeFiles/superpixel_crf.dir/depend:
	cd /home/xduser/LiHuan/superpixel_crf && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xduser/LiHuan/superpixel_crf /home/xduser/LiHuan/superpixel_crf /home/xduser/LiHuan/superpixel_crf /home/xduser/LiHuan/superpixel_crf /home/xduser/LiHuan/superpixel_crf/CMakeFiles/superpixel_crf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/superpixel_crf.dir/depend
