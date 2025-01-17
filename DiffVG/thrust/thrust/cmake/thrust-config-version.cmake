# Parse version information from version.h:
file(READ "${CMAKE_CURRENT_LIST_DIR}/../version.h" THRUST_VERSION_HEADER)
string(REGEX MATCH "#define[ \t]+THRUST_VERSION[ \t]+([0-9]+)" DUMMY "${THRUST_VERSION_HEADER}")
set(THRUST_VERSION_FLAT ${CMAKE_MATCH_1})
# Note that Thrust calls this the PATCH number, CMake calls it the TWEAK number:
string(REGEX MATCH "#define[ \t]+THRUST_PATCH_NUMBER[ \t]+([0-9]+)" DUMMY "${THRUST_VERSION_HEADER}")
set(THRUST_VERSION_TWEAK ${CMAKE_MATCH_1})

math(EXPR THRUST_VERSION_MAJOR "${THRUST_VERSION_FLAT} / 100000")
math(EXPR THRUST_VERSION_MINOR "(${THRUST_VERSION_FLAT} / 100) % 1000")
math(EXPR THRUST_VERSION_PATCH "${THRUST_VERSION_FLAT} % 100") # Thrust: "subminor" CMake: "patch"

# Build comparison versions:
set(THRUST_COMPAT "${THRUST_VERSION_MAJOR}.${THRUST_VERSION_MINOR}.${THRUST_VERSION_PATCH}")
set(THRUST_EXACT "${THRUST_COMPAT}.${THRUST_VERSION_TWEAK}")
set(FIND_COMPAT "${PACKAGE_FIND_VERSION_MAJOR}.${PACKAGE_FIND_VERSION_MINOR}.${PACKAGE_FIND_VERSION_PATCH}")
set(FIND_EXACT "${FIND_COMPAT}.${PACKAGE_FIND_VERSION_TWEAK}")

# Set default results
set(PACKAGE_VERSION ${THRUST_EXACT})
set(PACKAGE_VERSION_UNSUITABLE FALSE)
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)

# Test for compatibility (ignores tweak)
if (FIND_COMPAT VERSION_EQUAL THRUST_COMPAT)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()

# Test for exact (does not ignore tweak)
if (FIND_EXACT VERSION_EQUAL THRUST_EXACT)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
