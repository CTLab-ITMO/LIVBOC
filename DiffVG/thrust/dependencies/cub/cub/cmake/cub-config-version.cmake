# Parse version information from version.cuh:
file(READ "${CMAKE_CURRENT_LIST_DIR}/../version.cuh" CUB_VERSION_HEADER)
string(REGEX MATCH "#define[ \t]+CUB_VERSION[ \t]+([0-9]+)" DUMMY "${CUB_VERSION_HEADER}")
set(CUB_VERSION_FLAT ${CMAKE_MATCH_1})
# Note that CUB calls this the PATCH number, CMake calls it the TWEAK number:
string(REGEX MATCH "#define[ \t]+CUB_PATCH_NUMBER[ \t]+([0-9]+)" DUMMY "${CUB_VERSION_HEADER}")
set(CUB_VERSION_TWEAK ${CMAKE_MATCH_1})

math(EXPR CUB_VERSION_MAJOR "${CUB_VERSION_FLAT} / 100000")
math(EXPR CUB_VERSION_MINOR "(${CUB_VERSION_FLAT} / 100) % 1000")
math(EXPR CUB_VERSION_PATCH "${CUB_VERSION_FLAT} % 100") # CUB: "subminor" CMake: "patch"

# Build comparison versions:
set(CUB_COMPAT "${CUB_VERSION_MAJOR}.${CUB_VERSION_MINOR}.${CUB_VERSION_PATCH}")
set(CUB_EXACT "${CUB_COMPAT}.${CUB_VERSION_TWEAK}")
set(FIND_COMPAT "${PACKAGE_FIND_VERSION_MAJOR}.${PACKAGE_FIND_VERSION_MINOR}.${PACKAGE_FIND_VERSION_PATCH}")
set(FIND_EXACT "${FIND_COMPAT}.${PACKAGE_FIND_VERSION_TWEAK}")

# Set default results
set(PACKAGE_VERSION ${CUB_EXACT})
set(PACKAGE_VERSION_UNSUITABLE FALSE)
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)

# Test for compatibility (ignores tweak)
if (FIND_COMPAT VERSION_EQUAL CUB_COMPAT)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()

# Test for exact (does not ignore tweak)
if (FIND_EXACT VERSION_EQUAL CUB_EXACT)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
