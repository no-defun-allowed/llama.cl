;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: CL-USER -*-
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT

(uiop:define-package "LLAMA"
  (:use #:cl #:let-plus)
  (:import-from #:num-utils.arithmetic #:seq-max #:square)
  (:import-from #:array-operations #:partition #:sub #:argmax)
  #+lla (:import-from #:lla #:copy-array-from-memory #:create-array-from-memory #:vm!)
  ;; #+vml (:import-from #:vml #:v* #:v+)
  (:import-from #:alexandria+ #:unlessf)
  (:export #:read-checkpoint #:make-vocabulary #:generate))
