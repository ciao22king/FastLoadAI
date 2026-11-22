# Fedora / COPR

`fastloadai.spec` builds from the vendored source archive attached to the
GitHub release. The vendor directory is included, so COPR does not need
network access for Go modules.

## Build a source RPM

```bash
mkdir -p ~/rpmbuild/{SOURCES,SPECS}
curl -fL \
  -o ~/rpmbuild/SOURCES/fastloadai-1.1.0-vendor.tar.gz \
  https://github.com/ciao22king/FastLoadAI/releases/download/v1.1.0/fastloadai-1.1.0-vendor.tar.gz
cp fastloadai.spec ~/rpmbuild/SPECS/
rpmbuild -bs ~/rpmbuild/SPECS/fastloadai.spec
```

The result is `~/rpmbuild/SRPMS/fastloadai-1.1.0-1.src.rpm`.

## Build in COPR

Authenticate `copr-cli` with a Fedora API token, then:

```bash
copr-cli create fastloadai \
  --chroot fedora-rawhide-x86_64 \
  --chroot fedora-43-x86_64

copr-cli build fastloadai ~/rpmbuild/SRPMS/fastloadai-1.1.0-1.src.rpm
```

Add or remove chroots to match the Fedora releases you support. The package
does not bundle llama.cpp; install `llama-server` and `llama-quantize` from a
llama.cpp build on `PATH`.
