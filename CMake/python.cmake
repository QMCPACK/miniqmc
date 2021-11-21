# Support functions for handling python scripts

# Test whether a python modules is present
#   MODULE_NAME - input, name of module to test for
#   MODULE_PRESENT - output - True/False based on success of the import
function(TEST_PYTHON_MODULE MODULE_NAME MODULE_PRESENT)
  execute_process(
    COMMAND python ${miniqmc_SOURCE_DIR}/utils/test_import.py ${MODULE_NAME}
    OUTPUT_VARIABLE TMP_OUTPUT_VAR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${MODULE_PRESENT}
      ${TMP_OUTPUT_VAR}
      PARENT_SCOPE)
endfunction()

# Test python module prerequisites for a particular test script
#   module_list - input - list of module names
#   test_name - input - name of test (used for missing module message)
#   add_test - output - true if all modules are present, false otherwise
function(CHECK_PYTHON_REQS module_list test_name add_test)
  set(${add_test}
      true
      PARENT_SCOPE)
  foreach(python_module IN LISTS ${module_list})
    test_python_module(${python_module} has_python_module)
    if(NOT (has_python_module))
      message("Missing python module ${python_module}, not adding test ${test_name}")
      set(${add_test}
          false
          PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
