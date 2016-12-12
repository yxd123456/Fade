#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/home/wuhao/Project/caffe-android-lib/android_lib/boost/lib/libboost_system.a;/home/wuhao/Project/caffe-android-lib/android_lib/boost/lib/libboost_thread.a;/home/wuhao/Project/caffe-android-lib/android_lib/boost/lib/libboost_filesystem.a;/home/wuhao/Project/caffe-android-lib/android_lib/boost/lib/libboost_date_time.a;/home/wuhao/Project/caffe-android-lib/android_lib/boost/lib/libboost_atomic.a;/home/wuhao/Project/caffe-android-lib/android_lib/glog/lib/libglog.a;/home/wuhao/Project/caffe-android-lib/android_lib/gflags/lib/libgflags.a;/home/wuhao/Project/caffe-android-lib/android_lib/protobuf/lib/libprotobuf.a;/home/wuhao/Project/caffe-android-lib/android_lib/lmdb/lib/liblmdb.a;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;/home/wuhao/Project/caffe-android-lib/android_lib/openblas/lib/libopenblas.a"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.so"
  IMPORTED_SONAME_RELEASE "libcaffe.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.so" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
