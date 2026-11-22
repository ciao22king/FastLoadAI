#!/bin/sh
set -eu

VERSION=${1:?usage: $0 VERSION [OUTDIR]}
OUT=${2:-dist}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

export CGO_ENABLED=0
LDFLAGS="-s -w -X github.com/ciao22king/FastLoadAI/internal/tui.Version=$VERSION"

build_binary() {
	goos=$1
	goarch=$2
	dest=$3
	printf 'Building %s/%s\n' "$goos" "$goarch"
	(
		cd "$ROOT"
		GOOS=$goos GOARCH=$goarch CGO_ENABLED=0 \
			go build -trimpath -buildvcs=false -ldflags "$LDFLAGS" \
			-o "$dest" ./cmd/fastloadai
	)
}

make_tarball() {
	name=$1
	binary=$2
	stage="$TMP/$name"
	mkdir -p "$stage"
	install -m755 "$binary" "$stage/fastloadai"
	install -m644 "$ROOT/README.md" "$ROOT/LICENSE" "$stage/"
	tar -C "$stage" -czf "$OUT/$name.tar.gz" .
}

make_termux_deb() {
	arch=$1
	binary=$2
	# Termux calls the 64-bit x86 architecture x86_64; dpkg-deb accepts it
	# but warns because Debian uses amd64.
	stage="$TMP/termux-$arch"
	install -Dm755 "$binary" \
		"$stage/data/data/com.termux/files/usr/bin/fastloadai"
	mkdir -p "$stage/DEBIAN"
	cat > "$stage/DEBIAN/control" <<EOF
Package: fastloadai
Version: $VERSION
Architecture: $arch
Maintainer: ciao22king <268163239+ciao22king@users.noreply.github.com>
Installed-Size: $(du -sk "$stage" | cut -f1)
Depends: llama-cpp
Section: utils
Priority: optional
Homepage: https://github.com/ciao22king/FastLoadAI
Description: Hardware-aware local GGUF model runner and chat UI
 FastLoadAI downloads GGUF models, chooses a fitting quantization,
 and runs llama.cpp with a full-screen terminal chat.
EOF
	deb="$OUT/fastloadai_${VERSION}_${arch}.deb"
	dpkg-deb --root-owner-group --build "$stage" "$deb" >/dev/null
	printf '%s\n' "$deb"
}

make_debian_deb() {
	arch=$1
	binary=$2
	stage="$TMP/debian-$arch"
	install -Dm755 "$binary" "$stage/usr/bin/fastloadai"
	install -Dm644 "$ROOT/LICENSE" "$stage/usr/share/doc/fastloadai/copyright"
	install -Dm644 "$ROOT/README.md" "$stage/usr/share/doc/fastloadai/README.md"
	mkdir -p "$stage/DEBIAN"
	cat > "$stage/DEBIAN/control" <<EOF
Package: fastloadai
Version: $VERSION
Architecture: $arch
Maintainer: ciao22king <268163239+ciao22king@users.noreply.github.com>
Installed-Size: $(du -sk "$stage" | cut -f1)
Depends: ca-certificates
Suggests: llama.cpp
Section: utils
Priority: optional
Homepage: https://github.com/ciao22king/FastLoadAI
Description: Hardware-aware local GGUF model runner and chat UI
 FastLoadAI downloads GGUF models, chooses a fitting quantization,
 and runs llama.cpp with a full-screen terminal chat.
EOF
	deb="$OUT/fastloadai_${VERSION}_${arch}.deb"
	dpkg-deb --root-owner-group --build "$stage" "$deb" >/dev/null
	printf '%s\n' "$deb"
}

add_to_repo() {
	repo=$1
	deb=$2
	mkdir -p "$OUT/apt/$repo/pool/main/f/fastloadai"
	cp "$deb" "$OUT/apt/$repo/pool/main/f/fastloadai/"
}

finish_repo() {
	repo=$1
	arches=$2
	for arch in $arches; do
		dir="$OUT/apt/$repo/dists/stable/main/binary-$arch"
		mkdir -p "$dir"
		(
			cd "$OUT/apt/$repo"
			dpkg-scanpackages --arch "$arch" pool > "$dir/Packages" 2>/dev/null
		)
		gzip -9 -n -c "$dir/Packages" > "$dir/Packages.gz"
	done
	release="$TMP/$repo-Release"
	apt-ftparchive \
		-o APT::FTPArchive::Release::Origin=FastLoadAI \
		-o APT::FTPArchive::Release::Label=FastLoadAI \
		-o APT::FTPArchive::Release::Suite=stable \
		-o APT::FTPArchive::Release::Codename=stable \
		-o "APT::FTPArchive::Release::Architectures=$arches" \
		-o APT::FTPArchive::Release::Components=main \
		release "$OUT/apt/$repo/dists/stable" > "$release"
	mv "$release" "$OUT/apt/$repo/dists/stable/Release"
}

rm -rf "$OUT"
mkdir -p "$OUT"

build_binary android arm64 "$TMP/fastloadai-android-arm64"
build_binary linux amd64 "$TMP/fastloadai-linux-amd64"
build_binary linux arm64 "$TMP/fastloadai-linux-arm64"

make_tarball "fastloadai-$VERSION-linux-amd64" "$TMP/fastloadai-linux-amd64"
make_tarball "fastloadai-$VERSION-linux-arm64" "$TMP/fastloadai-linux-arm64"

termux_aarch64=$(make_termux_deb aarch64 "$TMP/fastloadai-android-arm64")
termux_x86_64=$(make_termux_deb x86_64 "$TMP/fastloadai-linux-amd64")
debian_amd64=$(make_debian_deb amd64 "$TMP/fastloadai-linux-amd64")
debian_arm64=$(make_debian_deb arm64 "$TMP/fastloadai-linux-arm64")

add_to_repo termux "$termux_aarch64"
add_to_repo termux "$termux_x86_64"
finish_repo termux "aarch64 x86_64"

add_to_repo debian "$debian_amd64"
add_to_repo debian "$debian_arm64"
finish_repo debian "amd64 arm64"

cat > "$OUT/apt/index.html" <<EOF
<!doctype html>
<meta charset="utf-8">
<title>FastLoadAI packages</title>
<h1>FastLoadAI packages</h1>
<p>Version $VERSION</p>
<h2>Termux</h2>
<p>Add the GitHub Pages APT repository:</p>
<pre>echo "deb [trusted=yes] https://ciao22king.github.io/FastLoadAI/termux stable main" \\
  &gt; \$PREFIX/etc/apt/sources.list.d/fastloadai.list
pkg update
pkg install fastloadai</pre>
<p>The TUR package is under review. Once accepted, it can be installed with
<code>pkg install tur-repo &amp;&amp; pkg install fastloadai</code>.</p>
<h2>Debian and Ubuntu</h2>
<pre>echo "deb [trusted=yes] https://ciao22king.github.io/FastLoadAI/debian stable main" \\
  | sudo tee /etc/apt/sources.list.d/fastloadai.list
sudo apt update
sudo apt install fastloadai</pre>
<p>Install <code>llama.cpp</code> separately so <code>llama-server</code> is on
<code>PATH</code>.</p>
EOF

(cd "$OUT" && sha256sum ./*.deb ./*.tar.gz > SHA256SUMS)

printf '\nPackages written to %s\n' "$OUT"
find "$OUT" -maxdepth 1 -type f -printf '  %f\n' | sort
