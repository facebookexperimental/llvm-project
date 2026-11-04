; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s

;; (S|U)MUL_LOHI lowering tests (T296600605).
;;
;; On NVPTX, v2i16 is a legal type and MULHS/MULHU on it are Expand, but
;; (S|U)MUL_LOHI used to fall through to default-Legal with no ISel
;; pattern. A vector divmod pair (quotient and remainder of the same
;; operands, e.g. `rows // 3` together with `rows % 3`) forms a
;; v2i16 (u|s)mul_lohi node, which fataled with
;; "Cannot select: v2i16 umul_lohi".
;;
;; v2i16 now expands to scalar lanes, and scalar i16/i32 (S|U)MUL_LOHI
;; lowers via mul.wide (NVPTXISD::MUL_WIDE_UNSIGNED/SIGNED), so the
;; divmod compiles (the dead low half folds to mul.hi). i64 stays
;; Expand (covered by arithmetic-int.ll).

define <2 x i16> @udivrem_v2i16_by3(<2 x i16> %x) {
; CHECK-LABEL: udivrem_v2i16_by3(
; CHECK: mul.hi.u16
; CHECK: mul.hi.u16
; CHECK: ret
  %q = udiv <2 x i16> %x, <i16 3, i16 3>
  %r = urem <2 x i16> %x, <i16 3, i16 3>
  %s = add <2 x i16> %q, %r
  ret <2 x i16> %s
}

define <2 x i16> @sdivrem_v2i16_by3(<2 x i16> %x) {
; CHECK-LABEL: sdivrem_v2i16_by3(
; CHECK: mul.hi.s16
; CHECK: mul.hi.s16
; CHECK: ret
  %q = sdiv <2 x i16> %x, <i16 3, i16 3>
  %r = srem <2 x i16> %x, <i16 3, i16 3>
  %s = add <2 x i16> %q, %r
  ret <2 x i16> %s
}
