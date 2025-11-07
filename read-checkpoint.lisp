;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: LLAMA -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT
(in-package #:llama)

;;; Duplicate part of the LLA API in case the user can't install LLA
#-lla
(defun copy-array-from-memory (array pointer internal-type)
  "Copy the memory area of type INTERNAL-TYPE at POINTER to ARRAY."
  (check-type array array)
  (let ((size (array-total-size array)))
    (cond
      ;; Handle single-float
      ((or (eq internal-type 'single-float)
           (eq internal-type 'short-float)
           (string-equal (symbol-name internal-type) "SINGLE"))
       (etypecase array
         ((simple-array single-float *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (sb-sys:sap-ref-single pointer (* index 4)))))
         ((simple-array * *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (coerce (sb-sys:sap-ref-single pointer (* index 4)) t))))))

      ;; Handle double-float
      ((or (eq internal-type 'double-float)
           (string-equal (symbol-name internal-type) "DOUBLE"))
       (etypecase array
         ((simple-array double-float *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (sb-sys:sap-ref-double pointer (* index 8)))))
         ((simple-array single-float *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (coerce (sb-sys:sap-ref-double pointer (* index 8)) 'single-float))))
         ((simple-array * *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (coerce (sb-sys:sap-ref-double pointer (* index 8)) t))))))

      ;; Handle 32-bit integer
      ((or (eq internal-type 'integer)
           (string-equal (symbol-name internal-type) "INTEGER"))
       (etypecase array
         ((simple-array (signed-byte 32) *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (sb-sys:signed-sap-ref-32 pointer (* index 4)))))
         ((simple-array * *)
          (loop for index below size do
            (setf (row-major-aref array index)
                  (coerce (sb-sys:signed-sap-ref-32 pointer (* index 4)) t))))))

      ;; Default case - assume single-float
      (t
       (loop for index below size do
         (setf (row-major-aref array index)
               (sb-sys:sap-ref-single pointer (* index 4)))))))
  (values))

#-lla
(defun create-array-from-memory (pointer internal-type dimensions
                                 &optional element-type)
  "Create an array from contents at POINTER."
  ;; Determine element type if not specified
  (let ((element-type (or element-type
                          (cond
                            ((or (eq internal-type 'single-float)
                                 (eq internal-type 'short-float)
                                 (string-equal (symbol-name internal-type) "SINGLE"))
                             'single-float)
                            ((or (eq internal-type 'double-float)
                                 (string-equal (symbol-name internal-type) "DOUBLE"))
                             'double-float)
                            ((or (eq internal-type 'integer)
                                 (string-equal (symbol-name internal-type) "INTEGER"))
                             '(signed-byte 32))
                            (t 'single-float)))))
    (let ((array (make-array dimensions :element-type element-type)))
      (copy-array-from-memory array pointer internal-type)
      array)))


(defun read-checkpoint (file)
  "Read model checkpoint using memory mapping for array creation"
  (mmap:with-mmap (addr fd size file)
    ;; Read config struct directly from memory
    (let* ((config-ptr addr)
           (dim (cffi:mem-ref config-ptr :int32 0))
           (hidden-dim (cffi:mem-ref config-ptr :int32 4))
           (num-layers (cffi:mem-ref config-ptr :int32 8))
           (num-heads (cffi:mem-ref config-ptr :int32 12))
           (num-kv-heads (cffi:mem-ref config-ptr :int32 16))
           (vocab-size (cffi:mem-ref config-ptr :int32 20))
           (sequence-len (cffi:mem-ref config-ptr :int32 24))

           ;; negative vocab size is hacky way of signaling unshared weights
	   (shared-weights (when (> vocab-size 0) t))
           (vocabulary-size (abs vocab-size))

           ;; Calculate dimensions
           (head-size (/ dim num-heads))
           (kv-dim (/ (* dim num-kv-heads) num-heads))

           ;; Create config structure
           (config (make-config :dim dim
                               :hidden-dim hidden-dim
                               :num-layers num-layers
                               :num-heads num-heads
                               :num-kv-heads num-kv-heads
                               :vocab-size vocab-size
                               :sequence-len sequence-len))

           ;; Start after config (7 int32s = 28 bytes)
           (weights-ptr (cffi:inc-pointer addr 28))
           (offset 0)

           (token-embedding-table)
           (rms-att-weight (make-array num-layers))
           (rms-ffn-weight (make-array num-layers))
           (wq (make-array num-layers))
           (wk (make-array num-layers))
           (wv (make-array num-layers))
           (wo (make-array num-layers))
           (w1 (make-array num-layers))
           (w2 (make-array num-layers))
           (w3 (make-array num-layers))
           (rms-final-weight)
           (wcls))

      (setf *config* config)

      ;; Helper to read array from current offset position
      (flet ((read-array (dimensions)
               (let* ((dims (if (listp dimensions) dimensions (list dimensions)))
                      (size (reduce #'* dims))
                      (current-ptr (cffi:inc-pointer weights-ptr offset))
                      (array (create-array-from-memory
                              current-ptr
                              #-lla 'short-float
                              #+lla lla::+single+
                              dims
                              'single-float)))
                 (incf offset (* size 4)) ; 4 bytes per float
                 array)))

        ;; Read all weights
        (setf token-embedding-table (read-array (list vocabulary-size dim)))

        ;; Read each weight type for all layers
        (loop for i from 0 below num-layers do
          (setf (aref rms-att-weight i) (read-array dim)))

        (loop for i from 0 below num-layers do
          (setf (aref wq i) (read-array (list dim dim))))

        (loop for i from 0 below num-layers do
          (setf (aref wk i) (read-array (list dim kv-dim))))

        (loop for i from 0 below num-layers do
          (setf (aref wv i) (read-array (list dim kv-dim))))

        (loop for i from 0 below num-layers do
          (setf (aref wo i) (read-array (list dim dim))))

        (loop for i from 0 below num-layers do
          (setf (aref rms-ffn-weight i) (read-array dim)))

        (loop for i from 0 below num-layers do
          (setf (aref w1 i) (read-array (list hidden-dim dim))))

        (loop for i from 0 below num-layers do
          (setf (aref w2 i) (read-array (list dim hidden-dim))))

        (loop for i from 0 below num-layers do
          (setf (aref w3 i) (read-array (list hidden-dim dim))))

        ;; Final RMS weight
        (setf rms-final-weight (read-array dim))

	;; Skip freq_cis_real and freq_cis_imag (for RoPE)
	;; Each is (sequence-len, head-size/2) floats
	(incf offset (* sequence-len head-size 4)) ; Total bytes: sequence_len * head_size * 4 bytes per float

        ;; Classifier weights (may share with token embeddings)
        (setf wcls (if shared-weights
                      token-embedding-table
                      (read-array (list vocabulary-size dim))))

        ;; Create transformer
        (make-transformer :config config
                         :weights (make-transformer-weights
                                  :token-embedding-table token-embedding-table
                                  :rms-att-weight rms-att-weight
                                  :rms-ffn-weight rms-ffn-weight
                                  :wq wq
                                  :wk wk
                                  :wv wv
                                  :wo wo
                                  :w1 w1
                                  :w2 w2
                                  :w3 w3
                                  :rms-final-weight rms-final-weight
                                  :wcls wcls)
                         :state (make-state config))))))
