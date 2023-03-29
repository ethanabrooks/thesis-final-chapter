{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      inherit (pkgs) mkShell poetry2nix lib;
      python = pkgs.python39;
      overrides = pyfinal: pyprev: rec {
        ipywidgets = pyprev.ipywidgets.overridePythonAttrs (old: {
          preferWheel = true;
        });

        # Use cuda-enabled jaxlib as required
        jaxlib = pyprev.jaxlibWithCuda.override {
          inherit (pyprev) absl-py flatbuffers numpy scipy six;
        };

        nbconvert = pyprev.nbconvert.overridePythonAttrs (old: {
          format = "pyproject";

          src = pyprev.fetchPypi {
            inherit (old) pname version;
            hash = "sha256-ju1nvYMU8+yHxDUcL2dK86BOWJCrkF1r2SfAWuwc8n0=";
          };

          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyprev.hatchling];
        });
        ray = pyprev.ray.overridePythonAttrs (old: {
          propagatedBuildInputs = (old.propagatedBuildInputs or []) ++ [pyfinal.pandas];
        });
        run-logger = pyprev.run-logger.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pyprev.poetry];
        });
        tensorflow-gpu =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.tensorflow-bin.overridePythonAttrs
          {inherit (pyprev.tensorflow-gpu) src version;};
        torch =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.pytorch-bin.overridePythonAttrs (old: {
            inherit (old) pname version;
            src = pkgs.fetchurl {
              url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
              sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
            };
          });
        torchdata = pyprev.torchdata.overridePythonAttrs (old: {
          format = "setuptools";
          src = pkgs.fetchgit {
            url = "https://github.com/pytorch/data.git";
            rev = "9eda5be3c6c6679ddbfd789f6e15928aa85e4833";
            sha256 = "sha256-HfwwRMYU44+pKEe5yPPS0I8s9zpNj6qJD2T2enx2m1Q=";
          };
        });
      };
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
      poetryEnv = poetry2nix.mkPoetryEnv poetryArgs;
      buildInputs = with pkgs; [
        alejandra
        poetry
        poetryEnv
      ];
    in rec {
      devShell = mkShell rec {
        inherit buildInputs;
        PYTHONBREAKPOINT = "ipdb.set_trace";
        LD_LIBRARY_PATH = "${pkgs.linuxPackages.nvidia_x11}/lib";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
