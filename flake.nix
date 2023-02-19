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
      inherit (pkgs) bash buildEnv cudaPackages dockerTools linuxPackages mkShell poetry2nix python39 stdenv;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      inherit (cudaPackages) cudatoolkit;
      inherit (linuxPackages) nvidia_x11;
      python = python39;
      overrides = pyfinal: pyprev: let
        inherit (pyprev) buildPythonPackage fetchPypi;
      in rec {
        # Use cuda-enabled jaxlib as required
        jaxlib = pyprev.jaxlibWithCuda.override {
          inherit (pyprev) absl-py flatbuffers numpy scipy six;
        };
        ray = pyprev.ray.overridePythonAttrs (old: {
          propagatedBuildInputs =
            (old.propagatedBuildInputs or [])
            ++ [pyfinal.pandas];
        });
        run-logger = pyprev.run-logger.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [pyprev.poetry];
        });
        tensorflow-gpu =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.tensorflow-bin.overridePythonAttrs
          {inherit (pyprev.tensorflow-gpu) src version;};
      };
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
      buildInputs = with pkgs; [
        alejandra
        coreutils
        nodePackages.prettier
        poetry
        poetryEnv
      ];
    in rec {
      devShell = mkShell rec {
        inherit buildInputs;
        PYTHONFAULTHANDLER = 1;
        PYTHONBREAKPOINT = "ipdb.set_trace";
        LD_LIBRARY_PATH = "${nvidia_x11}/lib";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
      packages.default = dockerTools.buildImage {
        name = "ppo";
        tag = "latest";
        copyToRoot =
          buildEnv
          {
            name = "image-root";
            pathsToLink = ["/bin" "/ppo"];
            paths = buildInputs ++ [pkgs.git ./.];
          };
        config = {
          Env = with pkgs; [
            "PYTHONFAULTHANDLER=1"
            "PYTHONBREAKPOINT=ipdb.set_trace"
            "LD_LIBRARY_PATH=/usr/lib64/"
            "PATH=/bin:$PATH"
          ];
          Cmd = ["${poetryEnv.python}/bin/python"];
        };
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
