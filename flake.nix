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
      inherit (pkgs) bash buildEnv cudaPackages dockerTools linuxPackages mkShell poetry2nix python39 stdenv fetchurl;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      inherit (cudaPackages) cudatoolkit;
      inherit (linuxPackages) nvidia_x11;
      python = python39;
      overrides = pyfinal: pyprev: let
        inherit (pyprev) buildPythonPackage fetchPypi;
      in rec {
        ipywidgets = buildPythonPackage rec {
          pname = "ipywidgets";
          version = "8.0.2";
          format = "setuptools";

          src = fetchPypi {
            inherit pname version;
            hash = "sha256-CMt1xuCpaDYUfL/cVVgK4E0T4F0m/7w3e04caLqiix8=";
          };

          propagatedBuildInputs = with pyfinal; [
            ipython
            ipykernel
            jupyterlab-widgets
            traitlets
            nbformat
            pytz
            widgetsnbextension
          ];

          checkInputs = with pyfinal; [pytestCheckHook];

          meta = {
            description = "IPython HTML widgets for Jupyter";
            homepage = "http://ipython.org/";
            license = pkgs.lib.licenses.bsd3;
            maintainers = with pkgs.lib.maintainers; [fridh];
          };

          doCheck = false;

          disabledTests = [
            "ipywidgets/widgets/tests/test_send_state.py"
            "ipywidgets/widgets/tests/test_set_state.py"
          ];
        };

        # Use cuda-enabled jaxlib as required
        jaxlib = pyprev.jaxlibWithCuda.override {
          inherit (pyprev) absl-py flatbuffers numpy scipy six;
        };

        nbconvert = let
          # see https://github.com/jupyter/nbconvert/issues/1896
          style-css = fetchurl {
            url = "https://cdn.jupyter.org/notebook/5.4.0/style/style.min.css";
            hash = "sha256-WGWmCfRDewRkvBIc1We2GQdOVAoFFaO4LyIvdk61HgE=";
          };
        in
          buildPythonPackage rec {
            pname = "nbconvert";
            version = "7.2.3";

            format = "pyproject";

            src = fetchPypi {
              inherit pname version;
              hash = "sha256-eufMxoSVtWXasVNFnufmUDmXCRPrEVBw2m4sZzzw6fg=";
            };

            # Add $out/share/jupyter to the list of paths that are used to search for
            # various exporter templates
            patches = [
              ./templates.patch
            ];

            postPatch = ''
              substituteAllInPlace ./nbconvert/exporters/templateexporter.py
              mkdir -p share/templates/classic/static
              cp ${style-css} share/templates/classic/static/style.css
            '';

            nativeBuildInputs = with pyfinal; [
              hatchling
            ];

            propagatedBuildInputs = with pyfinal;
              [
                beautifulsoup4
                bleach
                defusedxml
                jinja2
                #jupyter_core
                jupyterlab-pygments
                markupsafe
                mistune
                nbclient
                packaging
                pandocfilters
                pygments
                tinycss2
                traitlets
              ]
              ++ pkgs.lib.lists.optionals (pythonOlder "3.10") [
                importlib-metadata
              ];

            preCheck = ''
              export HOME=$(mktemp -d)
            '';

            checkInputs = with pyfinal; [
              ipywidgets
              pyppeteer
              pytestCheckHook
            ];
            doCheck = false;

            disabledTests = [
              # Attempts network access (Failed to establish a new connection: [Errno -3] Temporary failure in name resolution)
              "test_export"
              "test_webpdf_with_chromium"
              # ModuleNotFoundError: No module named 'nbconvert.tests'
              "test_convert_full_qualified_name"
              "test_post_processor"
            ];

            # Some of the tests use localhost networking.
            __darwinAllowLocalNetworking = true;
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
        torch =
          # Override the nixpkgs bin version instead of
          # poetry2nix version so that rpath is set correctly.
          pyprev.pytorch-bin.overridePythonAttrs (old: {
            inherit (old) pname version;
            src = fetchurl {
              url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
              sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
            };
          });
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
