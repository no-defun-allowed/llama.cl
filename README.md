<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

## llama.cl

This is a Common Lisp port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to idiomatic Common Lisp.

Why? Two reasons:

- Because Common Lisp is a fantastic language for experimentation, and this makes it easy to explore LLM techniques
- To serve as a reference implementation for the Common Lisp community

More than anything else it's the ease of AI experimentation, being able to mix expert systems, graphs, non-deterministic programming easily.

## How to run from emacs/slime/sly

### Prerequisites

We assume you have a working emacs, lisp and slime/sly setup.  Most of the systems `llama` requires are in [quicklisp](https://www.quicklisp.org/beta/), however [binary-types](https://github.com/snunez1/binary-types) is not in Quicklisp and you'll need to download it from the repository.  Put it in a location accessible to Quicklisp, like `~/common-lisp`.

1. Get the models from Karpathy's repo [(original instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) the dataset.

    ```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
    ```
2. Load the file `run.lisp` into an emacs buffer
3. Load slime with `M-x slime`
4. Load LLA with `(ql:quickload :lla)` (optional - requires setup)
5. Load LLAMA with `(ql:quickload :llama)` from the REPL
6. Move into the package `(in-package :llama)`
7. Initalise the system with `(init #P"stories15M.bin" #P"tokenizer.bin" 32000)` (adjust paths if neccessary)
8. Generate a story with: `(generate *model* *tokenizer*)`

You can experiment with temperature, prompts and various samplers.  See code for all the options.  Also tested and working with llama-2-7B.  You probably don't want to try anything larger unless you implement the CUDA kernels.

## Performance

My machine is running a 3.5 GHz 6-core Intel i7 5930, 256K/15MB cache with 64GB DDR4 RAM and with the `stories15M` I get about 2.5 tok/sec with CCL and 3.7 tok/s with SBCL.

If you want to use BLAS for matrix multiplication, you'll get about a 10X speed up.  Make sure that LLA is loaded _before_ you load `LLAMA`, if you do so it will automatically use the BLAS library.

Using LLA, the numbers are 14.4 tok/sec for CCL and 34.4 tok/sec for SBCL.

Interestingly, the parallel version (see the `forward` function) is slower on the the stories15M dataset.  Likely the parallisation overhead outweighs the benefits in this case.  I got the best results with lparallel kernel equal to the number of physical cores on the machine.


## Original README.md

For instructions on conversions to/from .bin format, training and other background, see the [original repo](https://github.com/karpathy/llama2.c)

