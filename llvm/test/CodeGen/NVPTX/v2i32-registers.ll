; RUN: llc < %s -mcpu=sm_100 -O0 -disable-post-ra -frame-pointer=all \
; RUN:   -verify-machineinstrs | FileCheck --check-prefixes=CHECK,NOV2I32 %s
; RUN: llc < %s -mcpu=sm_100 -O0 -disable-post-ra -frame-pointer=all \
; RUN:   -verify-machineinstrs -nvptx-v2i32 | \
; RUN:   FileCheck --check-prefixes=CHECK,V2I32 %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define <2 x i32> @test_add(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_add(
; V2I32:       .reg .b64
; NOV2I32-NOT: .reg .b64
; CHECK:       add.s32
  %r = add <2 x i32> %a, %b
  ret <2 x i32> %r
}

define <2 x float> @test_fadd(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: test_fadd(
; CHECK:       add.rn.f32x2
  %r = fadd <2 x float> %a, %b
  ret <2 x float> %r
}

; Vector loads/stores wider than 64 bits pack into v2i32 pairs, so the shape
; they are lowered to must follow the same predicate as the register class.
; Gating it on f32x2 alone asks for a v2i32 result that has no register class
; with v2i32 legalization off, which aborts with "Unhandled custom legalization".
define void @test_v4i32_ldst(ptr %in, ptr %out) {
; CHECK-LABEL: test_v4i32_ldst(
; CHECK:       ld.v4.b32
; CHECK:       st.v4.b32
  %v = load <4 x i32>, ptr %in, align 16
  %w = add <4 x i32> %v, %v
  store <4 x i32> %w, ptr %out, align 16
  ret void
}
