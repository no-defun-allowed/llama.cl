# LLAMA.CL

A Common Lisp implementation for Llama inference operations

[Report Bug](https://github.com/snunez1/llama.cl/issues) · [Request Feature](https://github.com/snunez1/llama.cl/issues)

## Table of Contents

1. About the Project
   - Objectives
   - Built With
2. Getting Started
   - Prerequisites
   - Installation
3. Usage
4. Performance
5. Roadmap
6. Contributing
7. License
8. Contact

## About the Project

LLAMA.CL is a Common Lisp implementation of Llama inference operations, designed for rapid experimentation, research, and as a reference implementation for the Common Lisp community. This project enables researchers and developers to explore LLM techniques within the Common Lisp ecosystem, leveraging the language's capabilities for interactive development and integration with symbolic AI systems.

### Objectives

- **Research-oriented interface**: Provide a platform for experimenting with LLM inference techniques in an interactive development environment.

- **Reference implementation**: Serve as a canonical example of implementing modern neural network inference in Common Lisp.

- **Integration capabilities**: Enable seamless combination with other AI paradigms available in Common Lisp, including expert systems, graph algorithms, and constraint-based programming.

- **Simplicity and clarity**: Maintain readable, idiomatic Common Lisp code that prioritizes understanding over premature optimization.

### Built With

- [num-utils](https://github.com/Lisp-Stat/num-utils)
- [array-operations](https://github.com/bendudson/array-operations)
- [alexandria](https://gitlab.common-lisp.net/alexandria/alexandria)
- [alexandria+](https://github.com/Symbolics/alexandria-plus)
- [let-plus](https://github.com/sharplispers/let-plus)
- [mmap](https://github.com/Shinmera/mmap)
- [anaphora](https://github.com/tokenrove/anaphora)

## Getting Started

### Prerequisites

LLAMA.CL requires:
- A Common Lisp implementation (currently SBCL-only as of version 0.0.5; pull requests for other implementations are welcome)
- Quicklisp or another ASDF-compatible system loader
- Pre-trained model weights in binary format

All dependencies are available through Quicklisp.

### Installation

#### Getting the source

1. Clone the repository to a location accessible to ASDF:
   ```bash
   cd ~/common-lisp
   git clone https://github.com/snunez1/llama.cl.git
   ```

2. Clear the ASDF source registry to recognize the new system:
   ```lisp
   (asdf:clear-source-registry)
   ```

#### Obtaining model weights

Download pre-trained models from [Karpathy's llama2.c repository](https://github.com/karpathy/llama2.c). For initial experimentation, the TinyStories models are recommended:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

#### Loading dependencies

Use Quicklisp to obtain required dependencies:
```lisp
(ql:quickload :llama)
```

## Usage

Initialize and generate text using the following workflow:

```lisp
;; Load the system
(ql:quickload :llama)

;; Switch to the LLAMA package
(in-package :llama)

;; Initialize with model and tokenizer
(init #P"stories15M.bin" #P"tokenizer.bin" 32000)

;; Generate text
(generate *model* *tokenizer*)
```

The system supports various generation parameters including temperature control, custom prompts, and different sampling strategies. Consult the source code for detailed parameter specifications.

The implementation has been validated with models up to llama-2-7B. Larger models may require additional optimization or hardware acceleration.

## Performance

### Lisp

On a reference system Intel(R) Core(TM) Ultra 7 155H 16/22 cores, 32GB DDR4 RAM), the stories110M model achieves approximately 3 tokens/second using SBCL and common lisp along and 22 tokens/sec with SBCL+LLA with 9 threads for lparallel and 3 for MKL BLAS.

Performance characteristics vary based on model size and hardware configuration. For the stories15M model, parallelization overhead may exceed benefits on some systems.  See the file benchmarks.md for benchmarking instructions.  You'll want to tune the lparallel and BLAS number of threads to find the sweet spot for you machine and model.


## Roadmap

- Extend compatibility to additional Common Lisp implementations
- Add support for quantized models

## Contributing

Contributions are welcome. Please submit pull requests for bug fixes, performance improvements, or additional Common Lisp implementation support. See the project's issue tracker for current priorities.

## License

Distributed under the MIT License. See LICENSE for more information.

## Contact

Project Link: [https://github.com/snunez1/llama.cl](https://github.com/snunez1/llama.cl)

