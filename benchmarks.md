## Threading Optimization Guide for LLaMA.cl

### Prerequisites

Before running any benchmarks, ensure you have the required systems loaded in the correct order:

```lisp
;; Load dependencies first
(asdf:load-system :lparallel)
(asdf:load-system :lla)
;; Load MKL/VML for optimization (if available)
(asdf:load-system :mkl/vml)
;; Then load the main system
(asdf:load-system :llama)
(in-package :llama)
```

**Note**: The LLA and MKL/VML systems provides BLAS and Intel Math Kernel Library optimizations for math operations. If you don't have BLAS or Intel MKL installed, you can skip loading `:mkl/vml` and and `:lla` the system will fall back to other implementations, but will run much slower.  Running with BLAS/MKL is about 10 times faster.

### Interactive REPL Testing

#### 1. Check Your System Configuration

Start by understanding your system:

```lisp
;; Check available cores
*system-cores*

;; View current threading state
(when lparallel:*kernel*
  (format t "Current lparallel threads: ~A~%" 
          (lparallel:kernel-worker-count lparallel:*kernel*)))
(format t "Current TBB threads: ~A~%" (uiop:getenv "TBB_NUM_THREADS"))

;; Check if MKL is loaded and active
(format t "MKL/VML loaded: ~A~%" (find-package :mkl/vml))
```

#### 2. Quick Performance Testing

Test a few sensible configurations quickly:

```lisp
;; Quick test with predefined configurations (5 runs each)
(quick-benchmark :runs 5)

;; Test specific configurations you're interested in
(quick-benchmark 
  :configs '((1 4)    ; 1 lparallel, 4 TBB
             (2 6)    ; 2 lparallel, 6 TBB  
             (4 4)    ; Balanced
             (8 2))   ; Heavy lparallel
  :runs 3)
```

#### 3. Comprehensive Analysis

For thorough optimization, run the full benchmark suite:

```lisp
;; Full benchmark respecting core limits (5 iterations per config)
(run-benchmarks 5 :respect-cores t)

;; Compare with over-subscription allowed (fewer iterations for speed)
(run-benchmarks 3 :respect-cores nil :max-cores 8)

;; Quick test with minimal iterations for fast feedback
(run-benchmarks 2 :respect-cores t :max-cores 6)
```

#### 4. Apply Optimal Configuration

Based on your benchmark results, set the best performing configuration:

```lisp
;; Example: if your analysis showed 2 lparallel, 6 TBB was optimal
(when lparallel:*kernel*
  (lparallel:end-kernel))
(setf lparallel:*kernel* (lparallel:make-kernel 2))
(set-tbb-threads 6)

;; Verify the configuration works
(time (generate *model* *tokenizer*))
```

#### 5. Interactive Testing Workflow

Here's a typical optimization session:

```lisp
;; Step 1: Load everything in correct order
(asdf:load-system :lparallel)
(asdf:load-system :lla) 
(asdf:load-system :mkl/vml)  ; For Intel MKL optimization
(asdf:load-system :llama)
(in-package :llama)

;; Step 2: Quick overview
*system-cores*
(quick-benchmark :runs 3)

;; Step 3: Focus on promising configurations
;; Adjust based on quick-benchmark results
(quick-benchmark 
  :configs '((1 6) (2 4) (3 3))  ; Example based on your results
  :runs 5)

;; Step 4: Full analysis of best candidates
(run-benchmarks 5 :respect-cores t :max-cores 8)

;; Step 5: Apply and test optimal settings
;; Use the lp/tbb values from your best result
(when lparallel:*kernel* (lparallel:end-kernel))
(setf lparallel:*kernel* (lparallel:make-kernel 2))  ; Adjust
(set-tbb-threads 6)  ; Adjust

;; Step 6: Verify performance
(time (generate *model* *tokenizer*))
```

#### 6. Monitoring and Comparison

Compare different approaches interactively:

```lisp
;; Test conservative threading
(when lparallel:*kernel* (lparallel:end-kernel))
(setf lparallel:*kernel* (lparallel:make-kernel 1))
(set-tbb-threads 4)
(time (dotimes (i 3) (generate *model* *tokenizer*)))

;; Test aggressive threading  
(when lparallel:*kernel* (lparallel:end-kernel))
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(set-tbb-threads 8)
(time (dotimes (i 3) (generate *model* *tokenizer*)))
```

### Key Guidelines

- **Start small**: Use `quick-benchmark` with few runs for initial exploration
- **System-aware**: The functions automatically detect your core count via `*system-cores*`
- **Incremental**: Test promising configurations with more iterations using `run-benchmarks`
- **Verify**: Always test your final configuration with actual model generation
- **MKL optimization**: Loading `:mkl/vml` can provide significant performance improvements for vector operations

### Expected Patterns

- **Lower core systems (≤8 cores)**: Often favor TBB-heavy configurations like `(2 6)`
- **Higher core systems (≥16 cores)**: May benefit from more lparallel threads like `(8 8)`
- **Memory-bound workloads**: Usually perform better with fewer total threads
- **CPU-bound workloads**: Can utilize more threads effectively
- **With MKL**: May see better performance with fewer threads due to more efficient vector operations

The benchmark functions will guide you to the optimal configuration by showing tokens/sec performance for each threading combination.  For example on my Intel(R) Core(TM) Ultra 7 155H (16/22) cores and running stories110M 9 lparallel and 3 TBB threads gave the best performance.

### Troubleshooting

If MKL/VML fails to load:
```lisp
;; Check if Intel MKL is properly installed and configured
(handler-case 
    (asdf:load-system :mkl/vml)
  (error (e) 
    (format t "MKL/VML not available: ~A~%" e)
    (format t "Continuing with standard math libraries...~%")))
```