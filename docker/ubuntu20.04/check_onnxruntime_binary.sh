#!/usr/bin/env bash
set -euo pipefail

# Quick self-check for onnxruntime binary in container:
# 1) debug symbols are present (.debug_* sections)
# 2) AVX-512 instructions are present in disassembly (zmm/k-mask usage)

DEFAULT_LIB="/opt/vcpkg_installed/x64-linux/lib/libonnxruntime.so"
LIB_PATH="${1:-$DEFAULT_LIB}"

if [[ ! -f "${LIB_PATH}" ]]; then
  echo "ERROR: libonnxruntime.so not found at: ${LIB_PATH}" >&2
  echo "Hint: pass an explicit path, e.g.:" >&2
  echo "  bash docker/ubuntu20.04/check_onnxruntime_binary.sh /path/to/libonnxruntime.so" >&2
  exit 2
fi

for bin in readelf objdump rg; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "ERROR: required tool not found: ${bin}" >&2
    exit 2
  fi
done

echo "Checking: ${LIB_PATH}"

echo
echo "[1/2] Debug symbols check"
if readelf -S "${LIB_PATH}" | rg -q '\.debug_(info|line|abbrev|str|ranges|aranges|loc)'; then
  echo "PASS: debug sections detected in libonnxruntime.so"
else
  echo "FAIL: no .debug_* sections found in libonnxruntime.so"
  echo "      build with ENABLE_ORT_DEBUG_SYMBOLS=ON"
  exit 1
fi

echo
echo "[2/2] AVX-512 instruction check"
AVX512_MATCHES="$(objdump -d "${LIB_PATH}" | rg -m 20 -o '\bzmm[0-9]+\b|\bk[0-7]\b' || true)"
if [[ -n "${AVX512_MATCHES}" ]]; then
  echo "PASS: AVX-512 instruction signatures detected (zmm/k-mask registers)"
  echo "Sample matches:"
  printf '%s\n' "${AVX512_MATCHES}" | awk 'NR<=20'
else
  echo "FAIL: no obvious AVX-512 signatures found in disassembly"
  echo "      build with ENABLE_AVX512=ON and ensure this is the rebuilt library"
  exit 1
fi

echo
echo "All checks passed."
