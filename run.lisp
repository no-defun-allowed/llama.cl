;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: LLAMA -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT
(in-package #:llama)

;;; Inference for Llama-2 Transformer model in Common Lisp

(defparameter *model* nil)
(defparameter *tokenizer* nil)
(defparameter *sampler* nil)


;;; Data structures
(defun print-config (config stream depth)
  "Print CONFIG structure readably"
  (declare (ignore depth))
  (print-unreadable-object (config stream :type t)
    (format stream "dim:~A hidden:~A layers:~A heads:~A kv-heads:~A vocab:~A seq:~A"
            (config-dim config)
            (config-hidden-dim config)
            (config-num-layers config)
            (config-num-heads config)
            (config-num-kv-heads config)
            (config-vocab-size config)
            (config-sequence-len config))))

(defstruct (config (:print-function print-config))
  "Model configuration parameters"
  (dim 0 :type fixnum)          ; transformer dimension
  (hidden-dim 0 :type fixnum)   ; for ffn layers
  (num-layers 0 :type fixnum)   ; number of layers (of encoder/decoder blocks)
  (num-heads 0 :type fixnum)    ; number of query heads
  (num-kv-heads 0 :type fixnum) ; number of key/value heads
  (vocab-size 0 :type fixnum)   ; vocabulary size, negative signals unshared weights
  (sequence-len 0 :type fixnum)) ; max sequence length

(defun print-transformer-weights (weights stream depth)
  "TRANSFORMER-WEIGHTS cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (weights stream :type t :identity t) ;let this signal an error if *print-readably* is T
    (princ "" stream)))

(defstruct (transformer-weights (:print-function print-transformer-weights))
  token-embedding-table	; vector of length vocab-size, with elements vector of (length dim)

  ;; weights for rmsnorms, vector of vectors
  rms-att-weight	;(layer, dim) rmsnorm weights
  rms-ffn-weight	;(layer, dim)

  ;; weights for matmuls. note dim == num-heads * head-size
  wq			;(layer, dim, number-heads * head-size)
  wk			;(layer, dim, number-kv-heads * head-size)
  wv			;(layer, dim, number-kv-heads * head-size)
  wo			;(layer, number-heads * head-size, dim)

  ;; weights for ffn
  w1			;(layer, hidden-dim, dim)
  w2			;(layer, dim, hidden-dim)
  w3			;(layer, hidden-dim, dim)

  rms-final-weight	;(dim,) final rmsnorm
  wcls)			;(optional) classifier weights for the logits, on the last layer

(defun make-state (config)
  "Allocate buffers for the run state
We technically don't need to do this here, but it may help the compiler generate more efficient code"
  (let+ (((&structure-r/o config- dim hidden-dim num-layers num-heads num-kv-heads vocab-size sequence-len) config))
    ;; (kv-dim (/ (* dim num-kv-heads) num-heads))) ;this was in Karpathy's code
    (make-run-state :x   (aops:zeros dim 'short-float)
		    :xb  (aops:zeros dim 'short-float)
		    :xb2 (aops:zeros dim 'short-float)
		    :hb  (aops:zeros hidden-dim 'short-float)
		    :hb2 (aops:zeros hidden-dim 'short-float)
		    :q (aops:zeros dim 'short-float)
		    :k (aops:zeros dim 'short-float)
		    :v (aops:zeros dim 'short-float)
		    :attention (aops:zeros (* num-heads sequence-len) 'short-float)
		    :logits    (aops:zeros (abs vocab-size) 'short-float)
		    :key-cache (aops:zeros `(,num-layers ,sequence-len ,dim) 'short-float)
		    :value-cache (aops:zeros `(,num-layers ,sequence-len ,dim) 'short-float))))

(defun print-run-state (run-state stream depth)
  "RUN-STATE cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (run-state stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (run-state (:print-function print-run-state))
  "Current wave of activations"
  x		;activation at current time stamp (dim,)
  xb		;same, but inside a residual branch (dim,)
  xb2		;an additional buffer just for convenience (dim,)
  hb		;buffer for hidden dimension in the ffn (hidden_dim,)
  hb2		;buffer for hidden dimension in the ffn (hidden_dim,)
  q		;query (dim,)
  k		;key (dim,)
  v		;value (dim,)
  attention	;buffer for scores/attention values (n_heads, seq_len)
  logits	;output logits

  ;; kv cache
  key-cache	;(layer, seq_len, dim)
  value-cache)	;(layer, seq_len, dim)

(defun print-transformer (transformer stream depth)
  "TRANSFORMER cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (transformer stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (transformer (:print-function print-transformer))
  config  ;the hyperparameters of the architecture (the blueprint)
  weights ;the weights of the model
  state   ;buffers for the "wave" of activations in the forward pass

  ;; some more state needed to properly clean up the memory mapping
  fd			 ;file descriptor for memory mapping
  data			 ;memory mapped data pointer
  file-size)		 ;size of the checkpoint file in bytes



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

      ;; Helper to read array from current offset position
      (flet ((read-array (dimensions)
               (let* ((dims (if (listp dimensions) dimensions (list dimensions)))
                      (size (reduce #'* dims))
                      (current-ptr (cffi:inc-pointer weights-ptr offset))
                      (array (create-array-from-memory
                              current-ptr
                              'short-float ;lla::+single+
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


;;; Matrix mathematics

;; If you have LLA loaded, then MM will be automatically imported from
;; there (see pkgdcl.lisp).  There is still room for optimisation with
;; LLA/BLAS.  For example having MM accept an existing output vector
;; instead of malloc and copy results as it does now.

;; If you don't have LLA loaded, the system will use the Common Lisp
;; version of matrix multiplication below.  See the discussions at:
;;   https://gist.github.com/mayerrobert/913b4c26103c614f9517360a4f00286a
;;   http://nklein.com/2009/06/speedy-matrix-multiplication-in-lisp-again/
;;   http://nklein.com/2009/06/trying-to-unconfound-lisp-speeds/
;;   The notes.org file in LLA
;;   http://tkpapp.blogspot.com/2010/05/upgraded-array-element-types-and-pinned.html
;; for ways to optimise this, including SB-SIMD operations.  There is
;; a lot of room for improvement in the Common Lisp matrix multiplication.
#+nil
(defun mm (x y)
  "Multiply vector X with matrix Y"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3)))
  (declare (type (simple-array single-float 1) x))
  (etypecase y
    ;; Suprisingly, all these perform about the same, despite the first having performance warnings from SBCL
    ;; ((simple-array single-float 1) (aops:vectorize-reduce #'+ (x y) (* x y))) ;3.8 tok/s
    ;; The reason to try MAP is so we can use LPARALLEL:PMAP to speed things up.  It doesn't work, PMAP is slower.
    ((simple-array single-float 1) (reduce #'+ (map 'vector ; 3.6,3.8 tok/s (single thread)
						    #'(lambda (x y)
							(declare (type short-float x y))
							(* x y))
						    x y)))
    (simple-array (aops:each-index* 'single-float (i)
		    (aops:sum-index j
		      (* (aref X j) (aref Y i j)))))))


;;; Neural network blocks, the dynamics of the transformer

(defun vm! (a b c)
  "Multiply vector A with matrix B and place results in C.
Returns: C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float *) b)
	   (type (simple-array single-float 1) a c))
  (aops:each-index! c (i)
    (aops:sum-index j
      (* (aref a j) (aref b i j))))
  c)

(defun rmsnorm (x w c)
  "Return the RMS norm of X and scale by weights W"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float *) w)
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

(defun softmax (x &optional (size (length x)))
  (let ((max-val (seq-max x))
	(sum 0.0))
    (declare (type single-float max-val sum))
    (loop for i below size
	  do (setf (aref x i) (exp (- (aref x i) max-val)))
	  summing (aref x i) into s
	  finally (setf sum s))
    (loop for i below size
          do (setf (aref x i) (/ (aref x i) sum))))
  x)

(defun v+ (a b c)
  "Destructive elementwise addition.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 1) a b c))
  (aops:each-index! c i
    (+ (aref a i) (aref b i)))
  c)

(defun v* (a b c)
  "Destructive elementwise multiplication.  Results are placed in C"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3))
	   (type (simple-array single-float 1) a b c))
  (aops:each-index! c i
    (* (aref a i) (aref b i)))
  c)


(defun forward (token-index position &key (transformer *model*))
  (let+ (((&structure transformer- config weights state) transformer)
	 ((&structure-r/o config- dim hidden-dim num-layers num-heads num-kv-heads vocab-size sequence-len) config)
	 ((&structure run-state- x xb xb2 hb hb2 q k v attention logits key-cache value-cache) state)
	 ((&structure transformer-weights-
		      token-embedding-table rms-att-weight rms-ffn-weight wq wk wv wo w1 w2 w3 rms-final-weight wcls)
	  weights)
	 (kv-dim (/ (* dim num-kv-heads) num-heads)) ;Multi Query Attention, see: https://arxiv.org/abs/1911.02150v1
	 ;; (kv-multiplier (/ num-heads num-kv-heads)) ;integer multiplier of the kv sharing in multiquery
	 (head-size (/ dim num-heads)))

    (replace x (aops:sub token-embedding-table token-index))
    (loop for layer below num-layers
	  do (rmsnorm x (aref rms-att-weight layer) xb)

	     ;; query, key and value matrix multiplications
	     (vm! xb (aref wq layer) q)
	     (vm! xb (aref wk layer) k)
	     (vm! xb (aref wv layer) v)

	     ;; RoPE relative positional encoding. See: https://arxiv.org/abs/2104.09864
	     ;; You'd think caching the frequency sin/cos vectors would be faster (HF does this), but apparently not:
	     ;; https://github.com/karpathy/llama2.c/issues/302
	     (loop for i below dim by 2
		   for head-dim = (mod i head-size)
		   for freq     = (/ (expt 10000f0 (/ head-dim head-size)))
		   for val      = (* position freq)
		   for fcr      = (cos val)
		   for fci      = (sin val)
		   for rotn     = (if (< i kv-dim) 2 1) ;how many vectors? 2 = q & k, 1 = q only
		   do (loop for v below rotn
			    for vec = (if (= v 0) q k) ;the vector to rotate, query or key
			    for v0  = (aref vec i)
			    for v1  = (aref vec (1+ i))
			    do (setf (aref vec i) (- (* v0 fcr) (* v1 fci))
				     (aref vec (1+ i)) (+ (* v0 fci) (* v1 fcr)))))

	     ;; Save key and value at this timestep (position) in cache
	     (setf (sub key-cache   layer position) k
		   (sub value-cache layer position) v)

	     ;; Multiquery attention, iterate over all heads
	     ;; TODO: conditionalise this on :lparallel in *features*
	     ;; (lparallel:pdotimes (head num-heads :discard num-heads)
	     (dotimes (head num-heads)
	       (let ((head-offset (* head sequence-len)))  ; Each head gets its own section of attention array
		 ;; Calculate attention scores for this head's section
		 (loop for timestep upto position
		       for sqrt-head-size = (sqrt head-size)
		       for head-q = (subseq q (* head head-size) (* (1+ head) head-size))
		       for head-k = (subseq
				     (sub key-cache layer timestep) (* head head-size) (* (1+ head) head-size))
		       do (setf (aref attention (+ head-offset timestep)) (/ ;dot product
									   (loop for q-elt :across head-q
										 for k-elt :across head-k
										 summing (* q-elt k-elt))
									   sqrt-head-size)))

		 ;; Create a displaced array view of just this head's attention section
		 (let ((head-attention (aops:displace attention (1+ position) head-offset)))
		   ;; Apply softmax to only this head's section
		   (setf (subseq attention head-offset (+ head-offset (1+ position))) (softmax head-attention (1+ position))))

		 ;; weighted sum of the values, store back into xb
		 (let ((xb (partition xb (* head head-size) (* (1+ head) head-size))))
		   (aops:each-index (i)
		     (setf (aref xb i) 0.0))
		   (loop for timestep upto position
			 for att = (aref attention (+ head-offset timestep))
			 for   v = (partition (sub value-cache layer timestep) (* head head-size) (* (1+ head) head-size))
			 do (loop for i below head-size
				  do (incf (aref xb i) (* att (aref v i))))))))

	     (vm! xb (aref wo layer) xb2) ;final matmul to get the output of the attention
	     (v+ x xb2 x)		  ;residual connection back into x
	     (rmsnorm x (aref rms-ffn-weight layer) xb) ;ffn rms norm
	     (vm! xb (aref w1 layer) hb)
	     (vm! xb (aref w3 layer) hb2)
	     (loop for i below hidden-dim
		   for val = (aref hb i)
		   do (setf (aref hb i) (* val (/ (1+ (exp (- val))))))) ;silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	     (v* hb hb2 hb)			       ;elementwise multiply with w3(x)
	     (vm! hb (aref w2 layer) xb)	       ;final matmul to get the output of the ffn
	     (v+ x xb x))			       ;residual connection
    ;; Layer loop ends above this line

    (rmsnorm x rms-final-weight x)   ;final rms norm
    (vm! x wcls logits)))            ;classifier into logits


;;; Tokenizer
(defun print-tokenizer (tokenizer stream depth)
  "TOKENIZER cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (tokenizer stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (tokenizer (:print-function print-tokenizer))
  vocabulary
  vocabulary-scores
  vocabulary-size
  max-token-length)

(defun make-vocabulary (file vocabulary-size)
  (let ((vocabulary        (make-array vocabulary-size :element-type 'string))
	(vocabulary-scores (make-array vocabulary-size :element-type 'float))
	max-token-length)
    (mmap:with-mmap (addr fd size file)
      (setf max-token-length (cffi:mem-ref addr :int))
      (loop for i below vocabulary-size
	    for ptr   = (cffi:inc-pointer addr 4) then (cffi:inc-pointer ptr (+ 4 4 count))
	    for score = (cffi:mem-ref ptr :float)
	    for count = (cffi:mem-ref ptr :int 4)
	    for token = (cffi:foreign-string-to-lisp ptr :offset 8 :count count)
	    do (setf (aref vocabulary i) token
		     (aref vocabulary-scores i) score)
	    finally (return (values vocabulary vocabulary-scores max-token-length))))))

(defun encode (text vocabulary scores)
  (let ((tokens (map 'vector (lambda (c) (position c vocabulary :test #'string=)) text)))
    (loop named outer
	  for best-score = -1e10
	  for best-id = -1
	  for best-index = -1
	  do (loop for i below (1- (length tokens))
		   for string = (concatenate 'string
					     (aref vocabulary (aref tokens i))
					     (aref vocabulary (aref tokens (1+ i))))
		   for id = (position string vocabulary :test #'string=)
		   if (and id (> (aref scores id) best-score)) ;This merge pair exists in vocabulary
		     do (setf best-score (aref scores id)
			      best-id id
			      best-index i))

	     (if (= best-index -1) (return-from outer tokens))
	     (setf (aref tokens best-index) best-id
		   tokens (concatenate 'vector (subseq tokens 0 (1+ best-index))
				       (subseq tokens (+ 2 best-index)))))))

;;; Sampler - greedy argmax, random, top-p, top-k
;;; Takes logits and returns a sampled token

(defun sample-mult (logits)
  (let ((r (random 1.0)))
    (loop for i below (length logits)
	  summing (aref logits i) into cdf
	  if (< r cdf) return i
	    finally (return (1- (length logits))))))

(defun sort-scores (scores predicate)
  "Returns an array of CONS, (index . score), sorted by score)."
  (let ((index -1))
    (sort (map 'vector (lambda (x)
			 (cons (incf index) x))
	       scores)
	  predicate :key #'cdr)))

;; I suspect that Karpathy's implementation takes the code path for
;; rounding errors.  Removing all scores below threshold, at least the
;; first time through, results in an empty set.
(defun sample-topp (logits p)
  (let* (;;(cutoff (/ (- 1.0 p) (- (length logits) 1)))	;values smaller than this cannot be part of the result
	 ;; (probabilities (sort-scores (remove-if #'(lambda (x) (< x cutoff)) logits) #'>)) ;remove smaller than cutoff and sort result
	 (probabilities (sort-scores logits #'>))
	 (r (random 1.0))
	 (last-index))

    (setf last-index (loop for i below (length probabilities)
			   summing (cdr (aref probabilities i)) into cumulative-probability
			   if (> cumulative-probability p) return i
			     finally (return  (1- (length logits)))))

    ;; Sample from our truncated sequence
    (loop for i below (length (subseq probabilities 0 last-index))
	  summing (cdr (aref probabilities i)) into cdf
	  if (< r cdf) return (car (aref probabilities i))
	    finally (return (car (aref probabilities last-index))))))

(defun sample (logits temperature &key topp topk)
  (declare (ignore topk))
  (if (< temperature short-float-epsilon)
      (argmax logits)
      (progn
	(setf logits (e/ logits temperature)
	      logits (softmax logits))
	(if (or (null topp) (<= topp 0) (>= topp 1))
	    (sample-mult logits)
	    (sample-topp logits topp)))))


;;; User API below

(defun init (model-path tokenizer-path &optional vocabulary-size) ;TODO: default to files in the repo
  "Initialise the model and tokenizer"
  (let+ (((&values vocabulary scores max-token-length) (make-vocabulary tokenizer-path vocabulary-size)))
    (setf *model*     (read-checkpoint model-path)
	  *tokenizer* (make-tokenizer :vocabulary vocabulary
				      :vocabulary-scores scores
				      :vocabulary-size vocabulary-size
				      :max-token-length max-token-length)))
  (values))


;;; Generation
(defun generate (model tokenizer &key
				   topp
				   (temperature 0.9)
				   (steps 256)
				   prompt)
  (let+ (((&structure tokenizer- vocabulary vocabulary-scores vocabulary-size max-token-length) tokenizer)
	 (token 1) next-token
	 (prompt-tokens (if prompt
			    (encode prompt vocabulary vocabulary-scores)
			    (vector 1))) ;default to BOS token
	 (start-time (get-internal-real-time)) end-time)

    (loop for position below steps
	  for logits = (forward token position :transformer model)
	  do (if (< position (length prompt-tokens))
		 (setf next-token (aref prompt-tokens position))
		 (setf next-token (sample logits temperature :topp topp)))
	     (format t "~A" (aref vocabulary next-token))
	     (setf token next-token))

    (setf end-time (get-internal-real-time))
    (let ((tok/s (float (/ steps (/ (- end-time start-time) internal-time-units-per-second)))))
      (if (< tok/s 1.0)
          (format t "~%sec/token: ~A~%" (/ tok/s))
          (format t "~%tokens/s: ~A~%" tok/s))
      tok/s)))
