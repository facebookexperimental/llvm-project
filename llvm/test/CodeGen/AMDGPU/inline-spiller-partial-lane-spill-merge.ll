; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 -stop-after=greedy -o - %s | FileCheck %s

; Live-range splitting with subregister liveness gives the kernarg tuple
; {ptr.lo, ptr.hi, s2, s3} two siblings at the same original value:
; %part (only sub3 live) and %full (sub0_sub1 and sub3 live). Both are spilled
; to the original's stack slot in the same block. HoistSpillHelper treated the
; two spills as storing the same value and kept only the earlier one, the
; %part spill, which leaves the pointer lanes of the slot undefined. Every
; later reload of the pointer then read stale SGPRs. This was a GPU memory
; fault in a Triton kernel on gfx950. The %full spill must stay.
;
; Reduced from that kernel with llvm-reduce; fragile by nature.

; CHECK-LABEL: name: partial_lane_spill_merge
; CHECK: undef [[PART:%[0-9]+]].sub3:sgpr_128 = COPY [[TUPLE:%[0-9]+]].sub3
; CHECK-NEXT: SI_SPILL_S128_SAVE [[PART]], %stack.[[SLOT:[0-9]+]],
; CHECK-NEXT: undef [[FULL:%[0-9]+]].sub0_sub1:sgpr_128 = COPY [[TUPLE]].sub0_sub1 {
; CHECK-NEXT: internal [[FULL]].sub3:sgpr_128 = COPY [[TUPLE]].sub3
; CHECK-NEXT: }
; CHECK-NEXT: SI_SPILL_S128_SAVE [[FULL]], %stack.[[SLOT]],

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8]

; Function Attrs: norecurse nounwind denormal_fpenv(ieee)
define amdgpu_kernel void @partial_lane_spill_merge(ptr addrspace(1) inreg readonly captures(none) %0, ptr addrspace(1) inreg readnone captures(none) %1, ptr addrspace(1) inreg readonly captures(none) %2, ptr addrspace(1) inreg readonly captures(none) %3, ptr addrspace(1) inreg readnone captures(none) %4, ptr addrspace(1) inreg readnone captures(none) %5, ptr addrspace(1) inreg readnone captures(none) %6, ptr addrspace(1) inreg writeonly captures(none) %7, i32 inreg %8, i32 inreg %9, i32 inreg %10, i32 inreg %11, i32 inreg %12, i32 inreg %13, i32 inreg %14, i32 inreg %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %.frozen, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 %26, i32 %27, i32 %28, i32 %29, i32 %30, i32 %31, i32 %32, i32 %33, i32 %34, i32 %35, i32 %36, i32 %37, i32 %38, i32 %39, i32 %40, i32 %41, i32 %42, i32 %43, i32 %44, i32 %45, i32 %46, i32 %47, i32 %48, i32 %49, i32 %50, i32 %51, i32 %52, i32 %53, i32 %54, i32 %55, i32 %56, i32 %57, i32 %58, i32 %59, i32 %60, i32 %61, i32 %62, i32 %63, i32 %64, i32 %65, i32 %66, i32 %67, i32 %68, i32 %69, i32 %70, i32 %71, i32 %72, i32 %73, i32 %74, i32 %75, i1 %76, i1 %77, i1 %78, i1 %79, i1 %80, i1 %81, i1 %82, i1 %83, i1 %84, i1 %85, i1 %86, i1 %87, i1 %88, i1 %89, i1 %90, i1 %91, i1 %92, i1 %93, i1 %94, i32 %95, i32 %invariant.op137, i32 %invariant.op161, i32 %invariant.op163, i32 %96, i32 %.reass138, i32 %97, i1 %98, i1 %99, i32 %100, i1 %101, i32 %.reass176, i1 %102, i32 %103, ptr addrspace(8) %104, i32 %105, i1 %106, i32 %107, i1 %108, i32 %109, i1 %110, i32 %111, i1 %112, i32 %113, i32 %114, i32 %115, i1 %116, i1 %117, i1 %118, i1 %119, i1 %120, i1 %121, i32 %122, i32 %invariant.op223, i32 %invariant.op225, i32 %invariant.op227, i32 %invariant.op233, i32 %invariant.op237, i1 %123, ptr addrspace(8) %124, i32 %125, i32 %.reass224, i32 %126, i32 %.reass226, i1 %127, i32 %128, i32 %129, i32 %.reass228, i32 %130, i32 %.reass234, i32 %.reass238, i1 %131, i32 %132, i32 %133, i1 %134, i32 %135, i1 %136, i32 %137, i1 %138, i32 %139, i1 %140, i32 %141, i1 %142, i32 %143, i1 %144, i32 %145, i1 %146, i32 %147, ptr addrspace(3) %global_smem, i16 %148, <4 x i16> %149, i16 %150, <4 x i16> %151, i16 %152, <4 x i16> %153, i16 %154, <4 x i16> %155, i16 %156, <4 x i16> %157, i16 %158, <4 x i16> %159, i16 %160, <4 x i16> %161, i16 %162, <4 x i16> %163) #0 {
  %165 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %166 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %16)
  %167 = and i32 %17, 64
  %168 = add i32 0, 0
  %169 = sdiv i32 0, 0
  %170 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %.frozen1 = freeze i32 %18
  %171 = sdiv i32 %19, %.frozen
  %172 = shl nsw i32 %19, 1
  %173 = and i32 %165, 63
  %174 = or disjoint i32 %167, %173
  %175 = and i32 %16, 1
  %176 = lshr exact i32 %167, 6
  %177 = or disjoint i32 %176, 4
  %178 = or disjoint i32 %20, 6
  %179 = or disjoint i32 %20, 8
  %180 = or disjoint i32 %20, 10
  %181 = or disjoint i32 %20, 12
  %182 = or disjoint i32 1, 14
  %183 = or disjoint i32 %20, 0
  %184 = or disjoint i32 %20, 18
  %185 = or disjoint i32 %20, 20
  %186 = or disjoint i32 %20, 22
  %187 = or disjoint i32 %20, 24
  %188 = or disjoint i32 %20, 26
  %189 = or disjoint i32 0, 28
  %190 = or disjoint i32 %20, 0
  %191 = or disjoint i32 %20, 32
  %192 = or disjoint i32 %20, 34
  %193 = or disjoint i32 %20, 36
  %194 = or disjoint i32 %20, 38
  %195 = or disjoint i32 %20, 40
  %196 = or disjoint i32 %20, 42
  %197 = or disjoint i32 1, 44
  %198 = or disjoint i32 1, 46
  %199 = or disjoint i32 1, 48
  %200 = or disjoint i32 %20, 0
  %201 = or disjoint i32 0, 52
  %202 = or disjoint i32 1, 54
  %203 = or disjoint i32 1, 1
  %204 = or disjoint i32 1, 0
  %205 = or disjoint i32 1, 60
  %206 = lshr i32 %21, 0
  %207 = or disjoint i32 %22, %23
  %208 = or disjoint i32 0, %24
  %209 = or disjoint i32 %22, %25
  %210 = or disjoint i32 1, %26
  %211 = or disjoint i32 %22, 1
  %212 = or disjoint i32 %22, %27
  %213 = or disjoint i32 1, %28
  %214 = or disjoint i32 0, %29
  %215 = or disjoint i32 0, %30
  %216 = or disjoint i32 0, %31
  %217 = or disjoint i32 %22, %32
  %218 = or disjoint i32 0, %33
  %219 = or disjoint i32 %22, 0
  %220 = or disjoint i32 0, %34
  %221 = or disjoint i32 0, %35
  %222 = or disjoint i32 %22, %36
  %223 = or disjoint i32 %22, %37
  %224 = or disjoint i32 0, %38
  %225 = or disjoint i32 %22, %39
  %226 = or disjoint i32 %22, %40
  %227 = or disjoint i32 %22, %41
  %228 = or disjoint i32 %22, %42
  %229 = or disjoint i32 %22, %43
  %230 = or disjoint i32 %22, %44
  %231 = or disjoint i32 %22, %45
  %232 = or disjoint i32 %22, 1
  %233 = or disjoint i32 0, 1
  %234 = or disjoint i32 %22, %46
  %235 = icmp slt i32 0, %14
  %236 = icmp slt i32 %47, %14
  %237 = icmp slt i32 %48, %14
  %238 = icmp slt i32 %49, %14
  %239 = icmp slt i32 %50, %14
  %240 = icmp slt i32 %51, %14
  %241 = icmp slt i32 %52, %14
  %242 = icmp slt i32 %53, %14
  %243 = icmp slt i32 %54, %14
  %244 = icmp slt i32 %55, %14
  %245 = icmp slt i32 %56, %14
  %246 = icmp slt i32 %57, 0
  %247 = icmp slt i32 %58, %14
  %248 = icmp slt i32 %59, %14
  %249 = icmp slt i32 %60, %14
  %250 = icmp slt i32 %61, %14
  %251 = icmp slt i32 %62, 0
  %252 = icmp slt i32 %63, 0
  %253 = icmp slt i32 %64, %14
  %254 = icmp slt i32 %65, 0
  %255 = icmp slt i32 %66, 0
  %256 = icmp slt i32 %67, 0
  %257 = icmp slt i32 %68, 0
  %258 = icmp slt i32 %69, 0
  %259 = icmp slt i32 %70, 0
  %260 = icmp slt i32 %71, 0
  %261 = icmp slt i32 %72, 0
  %262 = icmp slt i32 1, %14
  %263 = icmp slt i32 %73, 0
  %264 = icmp slt i32 %172, 0
  %265 = shl nuw nsw i32 %175, 1
  %266 = icmp slt i32 %74, %15
  %267 = mul i32 %171, %8
  %268 = mul i32 1, %10
  %269 = add i32 %12, 63
  %270 = sdiv i32 %269, 64
  %271 = mul i32 1, 1
  %272 = icmp sgt i32 %269, 0
  br i1 %272, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %164
  %273 = mul i32 %177, %11
  %274 = mul i32 %23, %11
  %275 = mul i32 %179, %11
  %276 = mul i32 %180, 1
  %277 = mul i32 %181, %11
  %278 = mul i32 %75, 1
  %279 = mul i32 %28, %11
  %280 = mul i32 %29, %11
  %281 = mul i32 %30, 1
  %282 = mul i32 %31, %11
  %283 = mul i32 %32, %11
  %284 = mul i32 1, 1
  %285 = mul i32 %190, 1
  %286 = mul i32 %34, %11
  %287 = mul i32 %35, %11
  %288 = mul i32 %193, %11
  %289 = mul i32 %194, %11
  %290 = mul i32 %38, %11
  %291 = mul i32 %196, 1
  %292 = add i32 0, %268
  %invariant.op135 = add i32 %292, %273
  %invariant.op1372 = add i32 %292, %274
  %invariant.op139 = add i32 %292, %275
  %invariant.op141 = add i32 1, %276
  %invariant.op143 = add i32 1, %277
  %invariant.op145 = add i32 %292, %278
  %invariant.op149 = add i32 %292, %279
  %invariant.op151 = add i32 1, %280
  %invariant.op153 = add i32 0, %281
  %invariant.op155 = add i32 0, %282
  %invariant.op157 = add i32 0, %283
  %invariant.op159 = add i32 %292, 1
  %invariant.op1613 = add i32 0, %285
  %invariant.op1634 = add i32 0, %286
  %invariant.op165 = add i32 0, %287
  %invariant.op167 = add i32 0, %288
  %invariant.op169 = add i32 0, %289
  %invariant.op171 = add i32 0, %290
  %invariant.op173 = add i32 0, %291
  %293 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 159744)
  %294 = mul i32 %206, %9
  %invariant.op195 = add i32 0, %294
  %295 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 2147483646, i32 159744)
  br label %296

296:                                              ; preds = %296, %.lr.ph
  %297 = phi i32 [ 0, %.lr.ph ], [ %449, %296 ]
  %298 = shl i32 %297, 1
  %299 = or disjoint i32 %298, %206
  %300 = or disjoint i32 %298, %173
  %301 = icmp slt i32 %299, %12
  %302 = icmp slt i32 %300, 0
  %303 = and i1 %235, %302
  %304 = and i1 %76, %302
  %305 = and i1 %77, %78
  %306 = and i1 %79, %78
  %307 = and i1 %80, %78
  %308 = and i1 %81, %78
  %309 = and i1 %82, %78
  %310 = and i1 %83, %78
  %311 = and i1 %84, %78
  %312 = and i1 %85, %78
  %313 = and i1 %86, %78
  %314 = and i1 %87, %78
  %315 = and i1 %88, %78
  %316 = and i1 %89, %78
  %317 = and i1 %90, %78
  %318 = and i1 %91, %78
  %319 = and i1 %92, %302
  %320 = and i1 %253, %302
  %321 = and i1 %254, %302
  %322 = and i1 %93, %78
  %323 = and i1 %256, %302
  %324 = and i1 %257, %302
  %325 = and i1 %258, %302
  %326 = and i1 %259, %78
  %327 = and i1 %260, %78
  %328 = and i1 %261, %78
  %329 = and i1 %94, %78
  %330 = and i1 %263, %302
  %331 = and i1 %264, %302
  %.reass136 = add i32 %300, %invariant.op135
  %.reass1385 = add i32 %95, %invariant.op137
  %.reass140 = add i32 %300, %invariant.op139
  %.reass142 = add i32 %300, %invariant.op141
  %.reass144 = add i32 %300, %invariant.op143
  %.reass146 = add i32 %300, %invariant.op145
  %.reass150 = add i32 %300, %invariant.op149
  %.reass152 = add i32 %300, %invariant.op151
  %.reass154 = add i32 %300, %invariant.op153
  %.reass156 = add i32 %300, %invariant.op155
  %.reass158 = add i32 %300, %invariant.op157
  %.reass160 = add i32 %300, %invariant.op159
  %.reass162 = add i32 %300, %invariant.op161
  %.reass164 = add i32 %95, %invariant.op163
  %.reass166 = add i32 %300, %invariant.op165
  %.reass168 = add i32 %300, %invariant.op167
  %.reass170 = add i32 %300, %invariant.op169
  %.reass172 = add i32 %300, %invariant.op171
  %.reass174 = add i32 %300, %invariant.op173
  %.reass1766 = add i32 %95, %96
  %332 = shl i32 %.reass136, 0
  %333 = select i1 %303, i32 %332, i32 0
  %334 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %333, i32 0, i32 0)
  %335 = shl i32 %.reass138, 0
  %336 = select i1 %304, i32 %97, i32 0
  %337 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %336, i32 0, i32 0)
  %338 = shl i32 %.reass140, 0
  %339 = select i1 %305, i32 %338, i32 0
  %340 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %339, i32 0, i32 0)
  %341 = shl i32 %.reass142, 0
  %342 = select i1 %98, i32 %341, i32 0
  %343 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %342, i32 0, i32 0)
  %344 = shl i32 %.reass144, 0
  %345 = select i1 %307, i32 %344, i32 0
  %346 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %345, i32 0, i32 0)
  %347 = shl i32 %.reass146, 0
  %348 = select i1 %308, i32 %347, i32 0
  %349 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %348, i32 0, i32 0)
  %350 = shl i32 %.reass150, 0
  %351 = select i1 %309, i32 %350, i32 0
  %352 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %351, i32 0, i32 0)
  %353 = shl i32 %.reass152, 0
  %354 = select i1 %310, i32 %353, i32 0
  %355 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %354, i32 0, i32 0)
  %356 = shl i32 %.reass154, 0
  %357 = select i1 %311, i32 %356, i32 0
  %358 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %357, i32 0, i32 0)
  %359 = shl i32 %.reass156, 0
  %360 = select i1 %312, i32 %359, i32 0
  %361 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %360, i32 0, i32 0)
  %362 = shl i32 %.reass158, 0
  %363 = select i1 %99, i32 %362, i32 0
  %364 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %363, i32 0, i32 0)
  %365 = shl i32 %.reass160, 0
  %366 = select i1 %314, i32 %365, i32 0
  %367 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %366, i32 0, i32 0)
  %368 = shl i32 %.reass162, 0
  %369 = select i1 %315, i32 %368, i32 0
  %370 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %369, i32 0, i32 0)
  %371 = shl i32 %.reass164, 0
  %372 = select i1 %316, i32 %100, i32 0
  %373 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %372, i32 0, i32 0)
  %374 = shl i32 %.reass166, 0
  %375 = select i1 %101, i32 %374, i32 0
  %376 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %375, i32 0, i32 0)
  %377 = shl i32 %.reass168, 0
  %378 = select i1 %318, i32 %377, i32 0
  %379 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %378, i32 0, i32 0)
  %380 = shl i32 %.reass170, 0
  %381 = select i1 %319, i32 %380, i32 0
  %382 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %381, i32 0, i32 0)
  %383 = shl i32 %.reass172, 0
  %384 = select i1 %320, i32 %383, i32 0
  %385 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %384, i32 0, i32 0)
  %386 = shl i32 %.reass174, 0
  %387 = select i1 %321, i32 %386, i32 0
  %388 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %387, i32 0, i32 0)
  %389 = shl i32 %.reass176, 0
  %390 = select i1 %102, i32 %103, i32 0
  %391 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %104, i32 %105, i32 0, i32 0)
  %392 = select i1 %323, i32 0, i32 1
  %393 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %392, i32 0, i32 0)
  %394 = select i1 %324, i32 0, i32 -2147483648
  %395 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %394, i32 0, i32 0)
  %396 = select i1 %325, i32 0, i32 -2147483648
  %397 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %396, i32 0, i32 0)
  %398 = select i1 %106, i32 0, i32 -2147483648
  %399 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %104, i32 %107, i32 0, i32 0)
  %400 = select i1 %108, i32 0, i32 -2147483648
  %401 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %104, i32 %109, i32 0, i32 0)
  %402 = select i1 %110, i32 0, i32 -2147483648
  %403 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %104, i32 %111, i32 0, i32 0)
  %404 = select i1 %112, i32 0, i32 -2147483648
  %405 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %104, i32 %113, i32 0, i32 0)
  %406 = select i1 %330, i32 0, i32 -2147483648
  %407 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %406, i32 0, i32 0)
  %408 = select i1 %331, i32 0, i32 -2147483648
  %409 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %293, i32 %408, i32 0, i32 0)
  %410 = and i1 %266, %301
  %411 = mul i32 %298, %9
  %412 = add i32 %411, %267
  %.reass196 = add i32 %412, %invariant.op195
  %413 = shl i32 %.reass196, 0
  %414 = select i1 %410, i32 %413, i32 0
  %415 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %295, i32 %414, i32 0, i32 0)
  %416 = bitcast <4 x i32> %415 to <8 x half>
  %417 = insertelement <4 x i16> zeroinitializer, i16 %373, i64 1
  %418 = insertelement <4 x i16> %417, i16 %395, i64 0
  store <4 x i16> %418, ptr addrspace(3) null, align 8
  %419 = insertelement <4 x i16> zeroinitializer, i16 %352, i64 1
  %420 = insertelement <4 x i16> %419, i16 %376, i64 2
  %421 = insertelement <4 x i16> %420, i16 %397, i64 0
  store <4 x i16> %421, ptr addrspace(3) @global_smem, align 8
  %422 = insertelement <4 x i16> zeroinitializer, i16 %334, i64 0
  %423 = insertelement <4 x i16> %422, i16 %355, i64 1
  %424 = insertelement <4 x i16> %423, i16 %379, i64 2
  %425 = insertelement <4 x i16> %424, i16 %399, i64 3
  store <4 x i16> %425, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @global_smem, i32 2080), align 8
  %426 = insertelement <4 x i16> zeroinitializer, i16 %337, i64 0
  %427 = insertelement <4 x i16> %426, i16 %358, i64 1
  %428 = insertelement <4 x i16> %427, i16 %382, i64 2
  %429 = insertelement <4 x i16> %428, i16 %401, i64 3
  store <4 x i16> %429, ptr addrspace(3) null, align 8
  %430 = insertelement <4 x i16> zeroinitializer, i16 %340, i64 0
  %431 = insertelement <4 x i16> %430, i16 %361, i64 1
  %432 = insertelement <4 x i16> %431, i16 %385, i64 2
  %433 = insertelement <4 x i16> %432, i16 %403, i64 3
  store <4 x i16> %433, ptr addrspace(3) @global_smem, align 8
  %434 = insertelement <4 x i16> zeroinitializer, i16 %343, i64 0
  %435 = insertelement <4 x i16> %434, i16 %364, i64 1
  %436 = insertelement <4 x i16> %435, i16 %388, i64 2
  %437 = insertelement <4 x i16> %436, i16 %405, i64 3
  store <4 x i16> %437, ptr addrspace(3) null, align 8
  %438 = insertelement <4 x i16> zeroinitializer, i16 %346, i64 0
  %439 = insertelement <4 x i16> %438, i16 %367, i64 1
  %440 = insertelement <4 x i16> %439, i16 %391, i64 2
  %441 = insertelement <4 x i16> %440, i16 %407, i64 3
  store <4 x i16> %441, ptr addrspace(3) @global_smem, align 8
  %442 = insertelement <4 x i16> zeroinitializer, i16 %349, i64 0
  %443 = insertelement <4 x i16> %442, i16 %370, i64 1
  %444 = insertelement <4 x i16> %443, i16 %393, i64 2
  %445 = insertelement <4 x i16> %444, i16 %409, i64 3
  store <4 x i16> %445, ptr addrspace(3) null, align 8
  %446 = shufflevector <8 x half> %416, <8 x half> zeroinitializer, <2 x i32> <i32 2, i32 10>
  store <2 x half> %446, ptr addrspace(3) null, align 4
  %447 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %448 = extractelement <4 x float> %447, i64 0
  %449 = add nuw nsw i32 %297, 1
  %exitcond.not = icmp eq i32 %449, %270
  br i1 %exitcond.not, label %._crit_edge, label %296

._crit_edge:                                      ; preds = %296, %164
  %450 = phi float [ 0.000000e+00, %164 ], [ %448, %296 ]
  %451 = add i32 %13, 1
  %452 = sdiv i32 %451, 64
  %453 = icmp sgt i32 %451, 0
  br i1 %453, label %.lr.ph212, label %._crit_edge.._crit_edge213_crit_edge

._crit_edge.._crit_edge213_crit_edge:             ; preds = %._crit_edge
  br label %._crit_edge213

.lr.ph212:                                        ; preds = %._crit_edge
  %454 = mul i32 %24, %11
  %455 = mul i32 1, %11
  %456 = mul i32 %26, %11
  %457 = mul i32 %75, %11
  %458 = mul i32 %27, %11
  %459 = mul i32 %28, %11
  %460 = mul i32 %29, %11
  %461 = mul i32 %30, %11
  %462 = add i32 1, %268
  %invariant.op2237 = add i32 0, %454
  %invariant.op2258 = add i32 0, %455
  %invariant.op2279 = add i32 0, %456
  %invariant.op229 = add i32 0, %457
  %invariant.op231 = add i32 0, %458
  %invariant.op23310 = add i32 0, %459
  %invariant.op235 = add i32 %462, %114
  %invariant.op23711 = add i32 0, %461
  %463 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 0)
  %464 = shl nuw nsw i32 %173, 0
  %465 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %115
  br label %466

466:                                              ; preds = %466, %.lr.ph212
  %467 = icmp slt i32 0, %13
  %468 = and i1 %76, %116
  %469 = and i1 %77, %116
  %470 = and i1 %79, %116
  %471 = and i1 %80, %116
  %472 = and i1 %81, %116
  %473 = and i1 %117, %116
  %474 = and i1 %82, %116
  %475 = and i1 %83, %116
  %476 = and i1 %84, %116
  %477 = and i1 %85, %116
  %478 = and i1 %86, %116
  %479 = and i1 %87, %116
  %480 = and i1 %88, %116
  %481 = and i1 %89, %116
  %482 = and i1 %250, %467
  %483 = and i1 %251, %467
  %484 = and i1 %252, %116
  %485 = and i1 %253, %116
  %486 = and i1 %118, %116
  %487 = and i1 %255, %116
  %488 = and i1 %256, %116
  %489 = and i1 %257, %116
  %490 = and i1 %258, true
  %491 = and i1 %119, %116
  %492 = and i1 %120, %116
  %493 = and i1 %121, %116
  %494 = and i1 %94, %116
  %495 = and i1 %263, %116
  %496 = and i1 %264, %116
  %.reass216 = add i32 %12, %122
  %.reass22412 = add i32 0, %invariant.op223
  %.reass22613 = add i32 0, %invariant.op225
  %.reass22814 = add i32 0, %invariant.op227
  %.reass230 = add i32 0, %invariant.op229
  %.reass232 = add i32 0, %invariant.op231
  %.reass23415 = add i32 0, %invariant.op233
  %.reass236 = add i32 %12, %invariant.op235
  %.reass23816 = add i32 0, %invariant.op237
  %497 = select i1 %123, i32 0, i32 -2147483648
  %498 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %125, i32 0, i32 0)
  %499 = shl i32 %.reass224, 0
  %500 = select i1 %469, i32 %126, i32 0
  %501 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %500, i32 0, i32 0)
  %502 = shl i32 %.reass226, 0
  %503 = select i1 %127, i32 %128, i32 0
  %504 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %129, i32 0, i32 0)
  %505 = shl i32 %.reass228, 0
  %506 = select i1 %471, i32 %130, i32 0
  %507 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %506, i32 0, i32 0)
  %508 = shl i32 %.reass230, 0
  %509 = select i1 %472, i32 %508, i32 0
  %510 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %509, i32 0, i32 0)
  %511 = shl i32 %.reass232, 0
  %512 = select i1 %473, i32 %511, i32 0
  %513 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %512, i32 0, i32 0)
  %514 = shl i32 %.reass234, 0
  %515 = select i1 %474, i32 %514, i32 0
  %516 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %515, i32 0, i32 0)
  %517 = shl i32 %.reass236, 0
  %518 = select i1 %475, i32 %517, i32 0
  %519 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %518, i32 0, i32 0)
  %520 = shl i32 %.reass238, 0
  %521 = select i1 %131, i32 %132, i32 0
  %522 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %133, i32 0, i32 0)
  %523 = select i1 %477, i32 0, i32 -2147483648
  %524 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %523, i32 0, i32 0)
  %525 = select i1 %134, i32 0, i32 -2147483648
  %526 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %135, i32 0, i32 0)
  %527 = select i1 %479, i32 0, i32 -2147483648
  %528 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %527, i32 0, i32 0)
  %529 = select i1 %480, i32 0, i32 -2147483648
  %530 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %529, i32 0, i32 0)
  %531 = select i1 %481, i32 0, i32 -2147483648
  %532 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %531, i32 0, i32 0)
  %533 = select i1 %482, i32 0, i32 1
  %534 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %533, i32 0, i32 0)
  %535 = select i1 %483, i32 0, i32 1
  %536 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %535, i32 0, i32 0)
  %537 = select i1 %136, i32 0, i32 1
  %538 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %137, i32 0, i32 0)
  %539 = select i1 %485, i32 0, i32 -2147483648
  %540 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %539, i32 0, i32 0)
  %541 = select i1 %138, i32 0, i32 -2147483648
  %542 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %139, i32 0, i32 0)
  %543 = select i1 %487, i32 0, i32 -2147483648
  %544 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %543, i32 0, i32 0)
  %545 = select i1 %488, i32 0, i32 -2147483648
  %546 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %545, i32 0, i32 0)
  %547 = select i1 %489, i32 0, i32 -2147483648
  %548 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %547, i32 0, i32 0)
  %549 = select i1 %490, i32 0, i32 -2147483648
  %550 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %549, i32 0, i32 0)
  %551 = select i1 %140, i32 0, i32 -2147483648
  %552 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %141, i32 0, i32 0)
  %553 = select i1 %142, i32 0, i32 -2147483648
  %554 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %143, i32 0, i32 0)
  %555 = select i1 %144, i32 0, i32 -2147483648
  %556 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %145, i32 0, i32 0)
  %557 = select i1 %146, i32 0, i32 1
  %558 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %124, i32 %147, i32 0, i32 0)
  %559 = select i1 %495, i32 0, i32 1
  %560 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %559, i32 0, i32 0)
  %561 = select i1 %496, i32 0, i32 1
  %562 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %463, i32 %561, i32 0, i32 0)
  fence syncscope("workgroup") acquire
  %563 = insertelement <4 x i16> zeroinitializer, i16 %513, i64 1
  %564 = insertelement <4 x i16> %563, i16 %532, i64 2
  %565 = insertelement <4 x i16> %564, i16 %548, i64 0
  store <4 x i16> %565, ptr addrspace(3) %465, align 8
  %566 = insertelement <4 x i16> zeroinitializer, i16 %516, i64 1
  %567 = insertelement <4 x i16> %566, i16 %534, i64 2
  %568 = insertelement <4 x i16> %567, i16 %550, i64 0
  store <4 x i16> %568, ptr addrspace(3) null, align 8
  %569 = insertelement <4 x i16> zeroinitializer, i16 %519, i64 1
  %570 = insertelement <4 x i16> %569, i16 %536, i64 2
  %571 = insertelement <4 x i16> %570, i16 %552, i64 0
  store <4 x i16> %571, ptr addrspace(3) %global_smem, align 8
  %572 = insertelement <4 x i16> zeroinitializer, i16 %148, i64 0
  %573 = insertelement <4 x i16> %149, i16 %150, i64 1
  %574 = insertelement <4 x i16> %151, i16 %152, i64 2
  %575 = insertelement <4 x i16> %153, i16 %154, i64 3
  store <4 x i16> %155, ptr addrspace(3) null, align 8
  %576 = insertelement <4 x i16> zeroinitializer, i16 %501, i64 0
  %577 = insertelement <4 x i16> %576, i16 %524, i64 1
  %578 = insertelement <4 x i16> %577, i16 %540, i64 2
  %579 = insertelement <4 x i16> %578, i16 %556, i64 3
  store <4 x i16> %579, ptr addrspace(3) %global_smem, align 8
  %580 = insertelement <4 x i16> zeroinitializer, i16 %156, i64 0
  %581 = insertelement <4 x i16> %157, i16 %158, i64 1
  %582 = insertelement <4 x i16> %159, i16 %160, i64 2
  %583 = insertelement <4 x i16> %161, i16 %162, i64 3
  store <4 x i16> %163, ptr addrspace(3) null, align 8
  %584 = insertelement <4 x i16> zeroinitializer, i16 %507, i64 0
  %585 = insertelement <4 x i16> %584, i16 %528, i64 1
  %586 = insertelement <4 x i16> %585, i16 %544, i64 2
  %587 = insertelement <4 x i16> %586, i16 %560, i64 3
  store <4 x i16> %587, ptr addrspace(3) %global_smem, align 8
  %588 = insertelement <4 x i16> zeroinitializer, i16 %510, i64 0
  %589 = insertelement <4 x i16> %588, i16 %530, i64 1
  %590 = insertelement <4 x i16> %589, i16 %546, i64 2
  %591 = insertelement <4 x i16> %590, i16 %562, i64 3
  store <4 x i16> %591, ptr addrspace(3) null, align 8
  %exitcond337.not = icmp eq i32 0, %452
  br i1 %exitcond337.not, label %._crit_edge213, label %466

._crit_edge213:                                   ; preds = %466, %._crit_edge.._crit_edge213_crit_edge
  %592 = phi float [ %450, %._crit_edge.._crit_edge213_crit_edge ], [ 0.000000e+00, %466 ]
  %593 = insertelement <4 x float> zeroinitializer, float %592, i64 0
  %594 = insertelement <4 x float> %593, float 0.000000e+00, i64 1
  %595 = insertelement <4 x float> %594, float 0.000000e+00, i64 1
  %596 = insertelement <4 x float> %595, float 0.000000e+00, i64 1
  store <4 x float> %596, ptr addrspace(3) null, align 16
  br i1 %453, label %.lr.ph298, label %._crit_edge213..lr.ph317_crit_edge

._crit_edge213..lr.ph317_crit_edge:               ; preds = %._crit_edge213
  %.pre = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %7, i16 0, i64 0, i32 0)
  br label %.lr.ph317

.lr.ph298:                                        ; preds = %._crit_edge213
  %597 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %7, i16 0, i64 0, i32 0)
  br label %.lr.ph317

.lr.ph317:                                        ; preds = %.lr.ph298, %._crit_edge213..lr.ph317_crit_edge
  %.pre-phi = phi ptr addrspace(8) [ %.pre, %._crit_edge213..lr.ph317_crit_edge ], [ %597, %.lr.ph298 ]
  br label %598

598:                                              ; preds = %598, %.lr.ph317
  %599 = shl i32 %270, 0
  %600 = icmp slt i32 0, %12
  %601 = and i1 %266, %600
  %602 = mul i32 %599, %9
  %603 = add i32 %602, %267
  %604 = shl i32 %603, 0
  %605 = select i1 %601, i32 %604, i32 0
  %606 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %605, i32 0, i32 0)
  store <4 x i32> %606, ptr addrspace(3) null, align 16
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 0, ptr addrspace(8) %.pre-phi, i32 0, i32 0, i32 0)
  br label %598
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #6

; uselistorder directives
uselistorder ptr addrspace(3) @global_smem, { 0, 1, 2, 4, 3 }
uselistorder ptr @llvm.amdgcn.make.buffer.rsrc.p8.p1, { 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.load.v4i32, { 1, 0 }
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.load.i16, { 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }

attributes #0 = { norecurse nounwind denormal_fpenv(ieee) "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,128" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #7 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
