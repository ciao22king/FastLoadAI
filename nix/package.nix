{
  lib,
  buildGoModule,
  fetchFromGitHub,
  version ? "1.1.0",
}:

buildGoModule {
  pname = "fastloadai";
  inherit version;

  src = fetchFromGitHub {
    owner = "ciao22king";
    repo = "FastLoadAI";
    rev = "v${version}";
    hash = "sha256-AOnn8eYZf/NK3OTEDq6eMN/Xqq+6AZ4nYih05uWCLrU=";
  };

  vendorHash = "sha256-8DVgOm8ly/t6nLCRJE2QZ4JpxePPAJEr//r5rEWyYao=";
  env.CGO_ENABLED = "0";
  subPackages = [ "cmd/fastloadai" ];

  ldflags = [
    "-s"
    "-w"
    "-X"
    "github.com/ciao22king/FastLoadAI/internal/tui.Version=${version}"
  ];

  doCheck = true;

  meta = {
    description = "Hardware-aware local GGUF model runner and chat UI for llama.cpp";
    homepage = "https://github.com/ciao22king/FastLoadAI";
    license = lib.licenses.asl20;
    mainProgram = "fastloadai";
    platforms = lib.platforms.linux;
  };
}
