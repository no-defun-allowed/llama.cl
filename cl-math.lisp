;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: LLAMA -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT
(in-package #:llama)

;;  See the discussions at:
;;   https://gist.github.com/mayerrobert/913b4c26103c614f9517360a4f00286a
;;   http://nklein.com/2009/06/speedy-matrix-multiplication-in-lisp-again/
;;   http://nklein.com/2009/06/trying-to-unconfound-lisp-speeds/
;;   The notes.org file in LLA
;;   http://tkpapp.blogspot.com/2010/05/upgraded-array-element-types-and-pinned.html
;; for ways to optimise matrix math, including SB-SIMD operations.  There is
;; a lot of room for improvement in the Common Lisp matrix multiplication.


;;; Neural network blocks, the dynamics of the transformer

(defmacro each-index! (vector index &body body)
  `(dotimes (,index (length ,vector))
     (setf (aref ,vector ,index) (progn ,@body))))

(defun vm! (a b c)
  "Multiply vector A with matrix B and place results in C.
Returns: C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 2) b)
	   (type (simple-array single-float 1) a c))
  (each-index! c i
    (let ((sum 0.0))
      (dotimes (j (length a) sum)
        (incf sum (* (aref a j) (aref b i j))))))
  c)

(defun rmsnorm (x w c)
  "Return the RMS norm of X and scale by weights W"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 1) w)
	   (type (simple-array single-float 1) x c))
  (let ((ss 0.0))				;sum of squares
    (declare (type single-float ss))
    (loop for x-elt :across x
          summing (square x-elt) into s
	  finally (setf ss s))
    (setf ss (/ ss (length x))
	  ss (+ ss 1e-5)
	  ss (/ (the single-float (sqrt ss))))
    (loop for i below (length x)
	  do (setf (aref c i) (* (aref w i) (* ss (aref x i)))))
    c))

(defun softmax (x &optional (start 0) (end (length x)))
  (declare (optimize speed)
           ((simple-array single-float 1) x))
  (let ((max-val most-negative-single-float)
	(sum 0.0))
    (declare (type single-float max-val sum))
    (loop for e across x do (setf max-val (max max-val e)))
    (loop for i from start below end
	  do (setf (aref x i) (exp (- (aref x i) max-val)))
	  summing (aref x i) into s
	  finally (setf sum s))
    (loop for i from start below end
          do (setf (aref x i) (/ (aref x i) sum))))
  x)

(defun v+ (a b c)
  "Destructive elementwise addition.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 1) a b c))
  (each-index! c i
    (+ (aref a i) (aref b i)))
  c)

(defun v* (a b c)
  "Destructive elementwise multiplication.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 1) a b c))
  (each-index! c i
    (* (aref a i) (aref b i)))
  c)

(defun v/ (a b c)
  "Destructive elementwise division.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
       (type (simple-array single-float 1) a b c))
  (each-index! c i
    (/ (aref a i) (aref b i)))
  c)

(defun v- (a b c)
  "Destructive elementwise subtraction.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
       (type (simple-array single-float 1) a b c))
  (each-index! c i
    (- (aref a i) (aref b i)))
  c)

(defun scale (alpha x)
  "Scale X by alpha.  X = X[1] * alpha"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
       (type (simple-array single-float 1) x)
       (type single-float alpha))
  (each-index! x i
    (* (aref x i) alpha))
  x)
