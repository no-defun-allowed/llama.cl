;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: CL-USER -*-
;;; Copyright (c) 2024, 2025 Steve Nunez
;;; SPDX-License-identifier: MIT

(uiop:define-package "LLAMA"
  (:use #:cl #:let-plus #:num-utils.elementwise)
  (:import-from #:num-utils.arithmetic #:sum #:seq-max #:square)
  (:import-from #:alexandria #:copy-array)
  (:import-from #:array-operations #:partition #:sub #:argmax)
  #+lla (:import-from #:lla #:copy-array-from-memory #:create-array-from-memory)
  (:import-from #:alexandria+ #:unlessf)
  (:export #:read-checkpoint #:make-vocabulary #:generate))
