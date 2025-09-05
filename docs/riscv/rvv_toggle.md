# RISC-V RVV Enable Toggle

This fork adds a build-time switch to explicitly enable or disable RISC-V Vector Extension (RVV 1.0) usage independent of the `--riscv_march` override.

## Option Summary

Command line flag (build.py): `--enable_riscv_v`
CMake cache variable: `onnxruntime_ENABLE_RISCV_V` (auto set from the flag)
Macro defined for C/C++: `ORT_RISCV_VECTOR_ENABLED` (1 when ON, 0 when OFF)

## Behavior

1. Default (no flag): RVV disabled unless you manually include `v` in `--riscv_march`.
2. With `--enable_riscv_v`:
   - If current effective `RISCV_MARCH_FLAGS` (after any `--riscv_march` override) does not contain `v`, the toolchain appends `v` to the `-march` string.
   - Defines `ORT_RISCV_VECTOR_ENABLED=1` for conditional compilation.
3. Without the flag: `ORT_RISCV_VECTOR_ENABLED=0` so RVV-specific intrinsic code can be excluded safely.

## Examples

Enable RVV using toolchain default baseline (`rv64gc` -> auto becomes `rv64gcv`):
```
python3 tools/ci_build/build.py --rv64 --riscv_toolchain_root /opt/riscv --riscv_qemu_path /usr/bin/qemu-riscv64 --enable_riscv_v --build_dir build/rvv_on --config Debug --update --build
```

Explicit march already containing `v` (no modification needed):
```
python3 tools/ci_build/build.py --rv64 --riscv_toolchain_root /opt/riscv --riscv_qemu_path /usr/bin/qemu-riscv64 --riscv_march rv64gcv_zba_zbb --enable_riscv_v --build_dir build/rvv_on --config Release --update --build
```

Force scalar-only build even if your default toolchain march has `v`:
```
python3 tools/ci_build/build.py --rv64 --riscv_toolchain_root /opt/riscv --riscv_qemu_path /usr/bin/qemu-riscv64 --riscv_march rv64gc --build_dir build/rvv_off --config Debug --update --build
```
(Do not pass `--enable_riscv_v`).

## Using the Macro

You can gate future RVV kernels like this:
```c++
#if ORT_RISCV_VECTOR_ENABLED
// RVV intrinsic accelerated implementation
#else
// Fallback scalar path
#endif
```

## Notes
- The toggle does not yet add any RVV intrinsic kernels; it only prepares the infrastructure.
- When adding vector code, prefer runtime detection later if heterogeneous deployment is required.
- Keep scalar fallbacks to maintain compatibility with cores lacking the V extension.
