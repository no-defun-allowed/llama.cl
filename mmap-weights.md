# SBCL Arrays and Foreign Memory: Why Zero-Copy is Impossible and How to Achieve Maximum Performance

## Part 1: Why Copying is Inevitable with SBCL Arrays

### The Fundamental Architecture Constraint

After extensive investigation of SBCL's array implementation, LLA's foreign memory interface, and various static memory approaches, we've definitively established that **creating true zero-copy Lisp arrays from memory-mapped files is impossible in SBCL**. Here's why:

#### 1. **SBCL's Array Memory Model**

SBCL arrays come in two flavors:
- **Simple arrays**: Header and data are contiguous in memory (header immediately followed by data)
- **Complex arrays**: Header contains a pointer to a simple array (not arbitrary memory)

```lisp
;; Simple array in memory:
[HEADER][DATA][DATA][DATA]...

;; Complex array in memory:
[HEADER] -> [SIMPLE-ARRAY-HEADER][DATA][DATA]...
```

There is no mechanism for an array header to point to arbitrary foreign memory.

#### 2. **Garbage Collector Invariants**

The SBCL garbage collector assumes:
- It allocated and owns all Lisp objects
- Array data is either inline (simple arrays) or points to other GC-managed objects
- It can move objects during collection (except in static space)
- All array memory can be traced and validated

Pointing an array header at mmap'd memory would violate these core assumptions and could cause crashes during GC.

#### 3. **Displaced Arrays Don't Help**

While displaced arrays can share storage, they can only displace to other Lisp arrays:
```lisp
(make-array dims :displaced-to another-lisp-array)
;; NOT: :displaced-to foreign-pointer
```

#### 4. **Static Vectors Don't Solve the Problem**

The `static-vectors` library and SBCL's `make-static-vector`:
- Allocate arrays in static (non-moving) memory
- Allow getting pointers TO arrays for foreign code
- Do NOT allow creating arrays FROM foreign pointers
- Still allocate header+data contiguously

#### 5. **No Low-Level Escape Hatch**

Even using SBCL internals like:
- `%array-data` 
- `set-array-header`
- `sb-kernel:make-array-header`

These all expect valid Lisp objects, not raw pointers. There's no supported way to construct just an array header pointing to foreign memory.

### Why This Design Makes Sense

SBCL's design prioritizes:
- **Safety**: Invalid memory access can't corrupt the Lisp heap
- **Performance**: Simple arrays have zero indirection overhead
- **GC Efficiency**: The collector knows exactly what memory it manages
- **Type Safety**: All array elements are valid Lisp objects

### The Verdict on Copying

For using standard Lisp arrays with mmap'd weight files:
```lisp
;; This ALWAYS copies data from mmap to Lisp heap
(lla::create-array-from-memory pointer lla::+single+ dimensions 'single-float)
```

**Copying is unavoidable**, but it's still better than reading from disk because:
- Memory mapping avoids OS file buffer copying
- The kernel pages in data as needed
- LLA uses optimized copying routines
- You get normal Lisp array semantics and safety

## Part 2: Maximum Performance Strategy Using SAPs and BLAS

### The Zero-Copy Solution: Direct BLAS with SAPs

Since we can't have Lisp arrays without copying, the maximum performance approach is to **bypass Lisp arrays entirely** for large weight matrices and work directly with System Area Pointers (SAPs).

### Implementation Strategy

#### 1. **Memory Map Weight File Once**
```lisp
(defstruct model-weights
  mmap-sap        ; SAP to start of mmap'd file
  mmap-size       ; Total file size
  config          ; Model configuration
  
  ;; SAPs to each weight region
  token-embedding-sap
  wq-saps         ; Array of SAPs, one per layer
  wk-saps
  wv-saps
  wo-saps
  w1-saps
  w2-saps
  w3-saps
  rms-att-saps
  rms-ffn-saps
  rms-final-sap
  wcls-sap)

(defun load-weights-as-saps (file)
  "Load model weights as SAPs - zero copy"
  (let* ((fd (sb-posix:open file sb-posix:o-rdonly))
         (size (sb-posix:lseek fd 0 sb-posix:seek-end))
         (_ (sb-posix:lseek fd 0 sb-posix:seek-set))
         (addr (sb-posix:mmap nil size 
                             sb-posix:prot-read
                             sb-posix:map-private fd 0))
         (sap (sb-sys:int-sap addr)))
    
    ;; Read config from SAP
    (let* ((dim (sb-sys:sap-ref-32 sap 0))
           (hidden-dim (sb-sys:sap-ref-32 sap 4))
           ;; ... read other config fields
           
           ;; Calculate offsets and create SAPs to each weight
           (offset 28)  ; After config
           (token-emb-sap (sb-sys:sap+ sap offset))
           ;; ... calculate other weight SAPs
           )
      
      (make-model-weights :mmap-sap sap
                          :token-embedding-sap token-emb-sap
                          ;; ... other weight SAPs
                          ))))
```

#### 2. **Direct BLAS Operations on SAPs**

BLAS functions can work directly with pointers, avoiding any copying:

```lisp
(defun matmul-sap (output-sap input-sap weight-sap m n k)
  "Matrix multiply using SAPs directly - zero copy"
  ;; Call BLAS SGEMM directly with SAPs
  (cffi:foreign-funcall "cblas_sgemm"
                        :int CblasRowMajor
                        :int CblasNoTrans
                        :int CblasNoTrans
                        :int m :int n :int k
                        :float 1.0            ; alpha
                        :pointer weight-sap   ; A
                        :int k               ; lda
                        :pointer input-sap    ; B
                        :int n               ; ldb
                        :float 0.0           ; beta
                        :pointer output-sap   ; C
                        :int n               ; ldc
                        :void))

(defun forward-sap (token position model-weights)
  "Forward pass using SAPs throughout"
  (let* ((config (model-weights-config model-weights))
         (dim (config-dim config))
         
         ;; Allocate small arrays for activations (these change each pass)
         (x (make-array dim :element-type 'single-float))
         (xb (make-array dim :element-type 'single-float))
         ;; ... other activation arrays
         
         ;; Get SAPs for activation arrays
         (x-sap (sb-sys:vector-sap x))
         (xb-sap (sb-sys:vector-sap xb)))
    
    ;; Copy token embedding (only this small vector)
    (let ((emb-offset (* token dim 4)))
      (copy-from-sap-to-array 
        (sb-sys:sap+ (model-weights-token-embedding-sap model-weights) emb-offset)
        x dim))
    
    ;; For each layer - use SAPs directly
    (loop for layer below (config-num-layers config) do
      ;; RMSnorm - small operation
      (rmsnorm-sap x-sap (aref (model-weights-rms-att-saps model-weights) layer) xb-sap dim)
      
      ;; Large matrix multiplies - zero copy!
      (matmul-sap q-sap xb-sap (aref (model-weights-wq-saps model-weights) layer) dim dim 1)
      (matmul-sap k-sap xb-sap (aref (model-weights-wk-saps model-weights) layer) kv-dim dim 1)
      (matmul-sap v-sap xb-sap (aref (model-weights-wv-saps model-weights) layer) kv-dim dim 1)
      
      ;; ... continue with attention and FFN using SAPs
      )))
```

#### 3. **Hybrid Approach for Best of Both Worlds**

```lisp
(defstruct transformer-hybrid
  ;; Weights stay as SAPs (zero-copy)
  weight-saps
  
  ;; Small activation arrays as Lisp arrays (for convenience)
  x xb xb2 hb hb2 q k v att logits
  
  ;; Cache arrays (need persistence across tokens)
  key-cache value-cache)

(defun forward-hybrid (transformer token position)
  "Use SAPs for weights, arrays for activations"
  (let* ((weights (transformer-hybrid-weight-saps transformer))
         (x (transformer-hybrid-x transformer))
         ;; Get SAP for activation when needed for BLAS
         (x-sap (sb-sys:vector-sap x)))
    
    ;; Copy only the small embedding vector
    (copy-embedding token weights x)
    
    ;; All large matrix operations use weight SAPs directly
    (blas-gemm-with-sap x-sap weight-sap result-sap ...)
    
    ;; Small operations can use Lisp arrays normally
    (rmsnorm x rms-weight xb)))
```

### Performance Benefits

This approach provides:

1. **Zero memory overhead**: Weights never copied from mmap
2. **Instant startup**: No gigabytes of copying at load time
3. **Direct BLAS calls**: No intermediate copying for matrix operations
4. **Cache efficiency**: OS can page out unused weights
5. **Memory sharing**: Multiple processes can share the same mmap'd file

### Practical Considerations

1. **SAP Lifetime**: Keep the mmap'd file descriptor open as long as SAPs are in use
2. **Alignment**: Ensure weight offsets are properly aligned for BLAS
3. **Endianness**: Handle byte order if needed
4. **Error Handling**: Check mmap success, handle unmapping on exit

### Example Weight Access Pattern

```lisp
;; Instead of: (aref weight-array i j)
;; Use:
(defun weight-ref (weight-sap dims i j element-type)
  (let* ((cols (second dims))
         (offset (* (+ (* i cols) j) 4))) ; 4 bytes for single-float
    (sb-sys:sap-ref-single weight-sap offset)))

;; For passing whole rows/matrices to BLAS:
(defun weight-row-sap (weight-sap dims row)
  (let* ((cols (second dims))
         (offset (* row cols 4)))
    (sb-sys:sap+ weight-sap offset)))
```

### Conclusion

For maximum performance with large language models in SBCL:
1. **Accept that Lisp arrays require copying** - it's a fundamental design constraint
2. **Use SAPs/pointers for large read-only weights** - achieving true zero-copy
3. **Use Lisp arrays for small activations** - maintaining code clarity where performance doesn't matter
4. **Call BLAS directly with SAPs** - avoiding all intermediate copying
5. **Keep the mmap'd file open** - ensuring SAPs remain valid

This hybrid approach gives you the best possible performance while working within SBCL's design constraints.
