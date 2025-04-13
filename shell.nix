{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name ="dev";
  buildInputs = [
  pkgs.python3
  pkgs.python312Packages.pip
  ];
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
   pkgs.stdenv.cc.cc.lib
   pkgs.libz
  ];
}
