find_package(Thrust REQUIRED CONFIG
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CMAKE_CURRENT_LIST_DIR}/.."
  COMPONENTS ${THRUST_REQUIRED_SYSTEMS}
  OPTIONAL_COMPONENTS ${THRUST_OPTIONAL_SYSTEMS}
)
