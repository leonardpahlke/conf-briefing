{
  description = "conf-briefing dev";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          name = "dev";

          buildInputs = with pkgs; [
            python312
            uv
            ruff
            glow
            just
            mdbook
            ollama
            chromium
            ffmpeg       # needed by scenedetect/opencv for video I/O
            tesseract    # OCR engine binary for pytesseract (fallback)
            # whisper-cpp: install separately with ROCm/Vulkan support
            # see: https://github.com/ggml-org/whisper.cpp#building
            # preflight auto-detects if whisper-cpp is on PATH
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.libxcb             # needed by opencv/scenedetect at import time
            pkgs.libx11
            pkgs.libGL              # needed by opencv (libGL.so.1)
            pkgs.glib               # needed by opencv (libgthread-2.0.so.0)
            pkgs.zstd.out           # needed by ROCm torch (libzstd.so.1)
          ];

          shellHook = ''
            # Skip uv sync when ROCm torch is installed — sync would overwrite
            # pip-installed packages (whisperx, huggingface-hub, etc.)
            if [ ! -f .rocm-torch ]; then
              uv sync --extra scrape --extra extract --quiet
            fi

            echo "--- Conf-Briefing Dev Shell ---"
            echo ""
            echo "Pipeline (pass event name, e.g. kubecon-eu-2026):"
            echo "  just collect <event>                  # scrape schedule + download videos"
            echo "  just extract-check <event>            # verify extract dependencies"
            echo "  just extract <event>                  # transcribe + slide OCR"
            echo "  just report <event>                   # clean → analyze → visualize → report"
            echo "  just query <event> \"question\"         # RAG Q&A (needs ollama)"
            echo ""
            echo "Dev:"
            echo "  just lint                             # ruff check + format"
            echo "  just fix                              # auto-fix lint issues"
            echo ""
            echo "Run 'just' to see all available commands."
          '';
        };
      });
    };
}
