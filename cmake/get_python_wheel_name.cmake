#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

function(get_python_wheel_name var)
    set(src)
    string(APPEND src "import sys, sysconfig; ")
    string(APPEND src "ver = '%d%d' % (sys.version_info.major, sys.version_info.minor); ")
    string(APPEND src "impl = 'cp%s' % ver; ")
    string(APPEND src "abi = impl ")
    string(APPEND src "+ ('u' if ver == '27' and sys.maxunicode == 0x10ffff else '') ")
    string(APPEND src "+ ('m' if sys.version_info[:2] <= (3, 7) else ''); ")
    string(APPEND src "plat = sysconfig.get_platform().replace('-', '_').replace('.', '_'); ")
    string(APPEND src "print('%s-%s-%s' % (impl, abi, plat), end=''); ")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${src}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE wheel_tag)
    if(NOT "${rc}" STREQUAL "0" OR "${wheel_tag}" STREQUAL "")
        message(FATAL_ERROR "Can not get Python wheel tag.")
    endif()
    execute_process(
        COMMAND bash "-c" "grep --color=never version pyproject.toml | grep --color=never -Eo '[0-9\.]+'"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE wheel_version
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${rc}" STREQUAL "0" OR "${wheel_version}" STREQUAL "")
        message(FATAL_ERROR "Can not get Python wheel version.")
    endif()
    set("${var}" "${CMAKE_PROJECT_NAME}-${wheel_version}-${wheel_tag}.whl" PARENT_SCOPE)
endfunction()
