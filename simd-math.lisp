(in-package #:llama)

(defun vm! (a b c)
  "Multiply vector A with matrix B and place results in C.
Returns: C"
  (declare (optimize (safety 0) (speed 3))
	   (type (simple-array single-float 2) b)
	   (type (simple-array single-float 1) a c))
  (each-index! c i
    (let ((sums (sb-simd-avx:f32.8 0.0)))
      (loop for j below (length a) by 8
            do (setf sums (sb-simd-avx:f32.8+
                           sums
                           (sb-simd-avx:f32.8*
                            (sb-simd-avx:f32.8-aref a j)
                            (sb-simd-avx:f32.8-aref b i j)))))
      (sb-simd-avx:f32.8-horizontal+ sums)))
  c)
