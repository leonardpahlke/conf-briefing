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
          ];

          shellHook = ''
            uv sync --quiet

            echo "--- Conf-Briefing Dev Shell ---"
            echo ""
            echo "Getting started:"
            echo "  1. Edit events/kubecon-eu-2026.toml              # configure event"
            echo "  2. just run                                      # run the full pipeline"
            echo ""
            echo "For RAG queries:"
            echo "  3. ollama pull nomic-embed-text                  # download embedding model"
            echo "  4. just index                                    # build vector index"
            echo "  5. just ask \"What are the main themes?\"          # query the data"
            echo ""
            echo "Each event gets its own .toml config and data dir under events/."
            echo "Run 'just' to see all available commands."
          '';
        };
      });
    };
}
