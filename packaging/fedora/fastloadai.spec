%global debug_package %{nil}

%global upstream https://github.com/ciao22king/FastLoadAI

Name:           fastloadai
Version:        1.1.0
Release:        1%{?dist}
Summary:        Hardware-aware local GGUF model runner and chat UI
License:        Apache-2.0
URL:            %{upstream}
Source0:        %{upstream}/releases/download/v%{version}/fastloadai-%{version}-vendor.tar.gz
BuildRequires:  golang >= 1.24

%description
FastLoadAI scans the local hardware, selects a compatible GGUF model, and
starts a tuned llama.cpp server with an interactive chat UI. It exposes
advanced llama.cpp cache, context, RoPE, SWA, and CPU affinity options.

%prep
%autosetup -n FastLoadAI-%{version}

%build
export CGO_ENABLED=0
go build -trimpath -buildvcs=false -mod=vendor -buildmode=pie \
  -ldflags "-s -w -X github.com/ciao22king/FastLoadAI/internal/tui.Version=%{version}" \
  -o _bin/%{name} ./cmd/%{name}

%check
go test -mod=vendor ./...

%install
install -Dpm0755 _bin/%{name} %{buildroot}%{_bindir}/%{name}

%files
%{_bindir}/%{name}
%license LICENSE
%doc README.md

%changelog
* Sun Nov 22 2026 FastLoadAI maintainers <maintainers@fastloadai.invalid> - 1.1.0-1
- Initial package.
