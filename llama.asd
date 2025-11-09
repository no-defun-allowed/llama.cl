;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: CL-USER -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024, 2025 Symbolics Pte Ltd
;;; SPDX-License-identifier: MIT

(defsystem "llama"
  :version "0.0.6"
  :license :MIT
  :author "Steve Nunez <steve@symbolics.tech>"
  :long-name   "Llama for Common Lisp"
  :description "Llama for Common Lisp"
  :long-description "A port of Karparty's llama2 inference code to Common Lisp"
  :source-control (:git "https://github.com/snunez1/llama.cl.git")
  :bug-tracker "https://github.com/snunez1/llama.cl/issues"
  :depends-on ("num-utils"
               "array-operations"
               "alexandria"
               "alexandria+"
               "let-plus"
               "mmap"
               "sb-simd")
  :components ((:file "pkgdcl")
	       #-lla (:file "cl-math")
	       #+lla (:file "lla-math")
               (:file "simd-math")
               (:file "run" :depends-on ("pkgdcl"))
               (:file "read-checkpoint")))
