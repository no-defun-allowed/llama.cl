;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: LLAMA -*-
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT
(in-package #:llama)

;; These functions aren't available directly via CFFI so we implement
;; them with BLAS.

;; Sadly, we have no way to optimise operations on displaced arrays.
;; See: https://groups.google.com/g/sbcl-help-archive/c/l3WXCHtxd1c

(defun v+ (a b c)
  "Destructive elementwise addition. Results are placed in C.

Optimized for the case where A and C are the same array (in-place operation)."
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array * (*)) a b c))
  ;; Size validation
  (let ((len-a (length a))
        (len-b (length b))
        (len-c (length c)))
    (declare (type fixnum len-a len-b len-c))
    (unless (and (>= len-a len-c) (>= len-b len-c))
      (error "Input arrays must be at least as large as output array"))

    #+vml (vml:add a b c)
    #-vml
    (if (eq a c)
	;; When a = c, just do c = c + b (no copy needed)
	(lla:axpy! 1.0 b c :n len-c)
	;; When a ≠ c, do c = a + b (copy then add)
	(progn
	  (lla:copy! a c :n len-c)
	  (lla:axpy! 1.0 b c :n len-c))))
    c)

(defun v* (a b c)
  "Destructive elementwise multiplication. Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array * (*)) a b c))
  ;; Size validation
  (let ((len-a (length a))
        (len-b (length b))
        (len-c (length c)))
    (declare (type fixnum len-a len-b len-c))
    (unless (and (>= len-a len-c) (>= len-b len-c))
      (error "Input arrays must be at least as large as output array"))

    #+vml (vml:mul a b c)
    #-vml
    (loop for i below len-c
          do (setf (aref c i) (* (aref a i) (aref b i)))))
  c)

;; Add after the existing functions, before rmsnorm

;;; Direct MKL VML interface for array slices
;;; Direct MKL VML interface for array slices
#+vml
(progn
  (cffi:defcfun ("vsExp" %vs-exp) :void
    (n :int)
    (a :pointer) 
    (y :pointer))
  
  (cffi:defcfun ("vdExp" %vd-exp) :void
    (n :int)
    (a :pointer)
    (y :pointer))

  (defun vml-exp-slice! (x start end)
    "Apply VML exp to array slice [start, end) in-place"
    (declare (type fixnum start end)
             (type (simple-array * (*)) x)
             (optimize (speed 3) (safety 0)))
    (let ((size (the fixnum (- end start))))
      (declare (type fixnum size))
      (when (> size 0)
        (cffi:with-pointer-to-vector-data (x-ptr x)
          (let ((slice-ptr (cffi:inc-pointer x-ptr 
                             (the fixnum 
                                  (* start 
                                     (the fixnum
                                          (etypecase x
                                            ((simple-array single-float (*)) 4)
                                            ((simple-array double-float (*)) 8))))))))
            (etypecase x
              ((simple-array single-float (*))
               (%vs-exp size slice-ptr slice-ptr))
              ((simple-array double-float (*))  
               (%vd-exp size slice-ptr slice-ptr)))))))))

(defun softmax (x &optional (start 0) (end (length x)))
  (declare (type (simple-array single-float (*)) x)
           (type fixnum start end)
           (optimize (speed 3) (safety 0)))
  (let ((max-val 0.0)
        (sum 0.0)
        (size (- end start)))
    (declare (type single-float max-val sum)
             (type fixnum size))

    ;; Step 1: Find max value efficiently (inlined)
    ;; Note: cblas_isamax finds max |x[i]|, not max x[i], so we use optimized loop
    (if (>= start end)
        (setf max-val 0.0)
        (progn
          (setf max-val (aref x start))
          (loop for i fixnum from (1+ start) below end
                do (when (> (aref x i) max-val)
                     (setf max-val (aref x i))))))

    ;; Step 2: Subtract max_val in-place
    (loop for i fixnum from start below end
          do (decf (aref x i) max-val))

    ;; Step 3: Vectorized exponential using direct VML slice operation
    #+vml
    (vml-exp-slice! x start end)

    #-vml
    (loop for i fixnum from start below end
          do (setf (aref x i) (the single-float (exp (aref x i)))))

    ;; Step 4 & 5: Sum and normalize using BLAS with pointer arithmetic
    (cffi:with-pointer-to-vector-data (x-ptr x)
      (let ((range-ptr (cffi:inc-pointer x-ptr (* start 4))))
        ;; Step 4: Sum using BLAS ASUM with pointer offset
        (setf sum (cffi:foreign-funcall "cblas_sasum"
                                        :int32 size
                                        :pointer range-ptr
                                        :int32 1
                                        :float))

        ;; Step 5: Normalize using BLAS SCAL with pointer offset
        (cffi:foreign-funcall "cblas_sscal"
                              :int32 size
                              :float (/ sum)
                              :pointer range-ptr
                              :int32 1
                              :void)))
    x))



;;; Neural network blocks, the dynamics of the transformer
(defun rmsnorm (x w c)
  "Return the RMS norm of X and scale by weights W using BLAS operations"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
       (type (simple-array single-float *) w)
       (type (simple-array single-float 1) x c))
  (let* ((n (length x))
         (ss (lla:dot x x :n n)))
    (declare (type single-float ss))
    ;; Compute normalization factor
    (setf ss (/ ss n)
      ss (+ ss 1e-5)
      ss (/ (the single-float (sqrt ss))))
    ;; Possibly copy x to c, then scale by ss using BLAS SCAL
    (unless (eq x c)
      (lla:copy! x c :n n))
    (lla:scal! ss c :n n)
    ;; Element-wise multiplication with weights
    #+vml
    (vml:v* c w c)  ; c = c .* w (in-place)
    #-vml
    (loop for i below n
      do (setf (aref c i) (* (aref c i) (aref w i))))
    c))

