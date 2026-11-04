; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
; RUN:   -gfx1250-regalloc-time-warn-threshold-seconds=0 -o - %s 2>&1 \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
; RUN:   -gfx1250-regalloc-time-warn-threshold-seconds=0 -o - %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-WARN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
; RUN:   -gfx1250-regalloc-time-warn=false \
; RUN:   -gfx1250-regalloc-time-warn-threshold-seconds=0 -o - %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-WARN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-WARN

; CHECK: warning: register allocation for function 'slow_gfx1250_regalloc' on gfx1250 took {{[0-9]+}}s, exceeding the 0s warning threshold. gfx1250 has many more registers than previous AMD GPUs, and optimizing register-allocator compile time for gfx1250 is a work in progress
; NO-WARN-NOT: register allocation for function 'slow_gfx1250_regalloc' on gfx1250 took

define amdgpu_kernel void @slow_gfx1250_regalloc(ptr addrspace(1) %out, i32 %a, i32 %b) {
entry:
  %x0 = add i32 %a, %b
  %x1 = mul i32 %x0, %a
  %x2 = add i32 %x1, %b
  %x3 = mul i32 %x2, %x0
  %x4 = add i32 %x3, %x1
  %x5 = mul i32 %x4, %x2
  %x6 = add i32 %x5, %x3
  %x7 = mul i32 %x6, %x4
  store i32 %x7, ptr addrspace(1) %out, align 4
  ret void
}
