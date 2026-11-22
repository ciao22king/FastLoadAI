# Termux

Copy the package recipe into a checkout of
[termux-packages](https://github.com/termux/termux-packages):

```bash
cp -a packaging/termux/fastloadai /path/to/termux-packages/packages/
cd /path/to/termux-packages
./build-package.sh fastloadai
```

For a Docker build from a non-Android host:

```bash
./scripts/run-docker.sh ./build-package.sh fastloadai
```

The package depends on `llama-cpp`, which provides `llama-server` and
`llama-quantize`. It is limited to 64-bit Termux because `llama-cpp` does not
support 32-bit Android.
