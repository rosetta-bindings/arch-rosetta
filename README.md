<h1 align="center">🧊 Arch&nbsp;×&nbsp;Rosetta</h1>

<p align="center">
  <em>Automatic, multi-language bindings for Arch, a 3D BEM code.</em>
</p>

<p align="center">
  <a href="https://github.com/xaliphostes/arch3"><img src="https://img.shields.io/badge/library-ARCH-2c7fb8.svg" alt="Arch"></a>
  <a href="https://github.com/xaliphostes/rosetta"><img src="https://img.shields.io/badge/generator-Rosetta-6a3d9a.svg" alt="Rosetta"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-26%20(P2996)-blue.svg?logo=cplusplus" alt="C++26">
  <img src="https://img.shields.io/badge/bindings-Python%20%7C%20Node%20%7C%20…-green.svg" alt="Bindings">
</p>

---

This project points [**Rosetta**](https://github.com/xaliphostes/rosetta) at the [**Arch Library**](https://github.com/xaliphostes/arch3.git) and, from a single [`manifest.json`](manifest.json), generates ready-to-build bindings **without touching a line of Arch's source**.

### Fetch Arch and Rosetta

```sh
cmake -S . -B build && cmake --build build -j
```

### Generate the generator for PMP

```sh
extern/rosetta/bin/rosetta_gen manifest.json gen
cmake -S gen -B gen/build && cmake --build gen/build
```

### Generate the bindings

```sh
./generator bindings
```

### Compile the bindings

#### 1. Example for Python:
```sh
cmake -S bindings/python -B bindings/python/build && cmake --build bindings/python/build -j
```

#### 2. Example for nodejs:
```sh
cd bindings/node && npm i && npm run build && cd ../..
```

#### 3. Example for WebAssembly

First set up a stock [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (no fork needed):
```sh
# One-time emsdk setup
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh && cd ..
```

Then configure with `emcmake`:
```sh
# Build the wasm bindings (emits pmp.js + pmp.wasm)
emcmake cmake -S bindings/wasm-expanded -B bindings/wasm-expanded/build
cmake --build bindings/wasm-expanded/build -j
```

### Run the examples

Each backend has a ready-to-run demo at the project root. Build the matching
binding above first, then run from the project root:

#### 1. Python

```sh
python3 example_python.py
```

#### 2. Node (native N-API addon)

```sh
node example_node.js
```

#### 3. WebAssembly with Node

```sh
node example_wasm.js
```
