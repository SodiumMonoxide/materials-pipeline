#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash install.sh [--prefix PATH]

Install Materials Pipeline and both terminal commands:
  materials-pipeline
  materials-pipline

The default prefix is ~/.local. Commands go in PREFIX/bin and the
application goes in PREFIX/share/materials-pipeline. The installer
requires Linux, Python 3.10+, Bash, coreutils, and util-linux (flock).
Configure VASP, its PAW data, and optional Python dependencies separately.
EOF
}

die() {
    printf 'install: %s\n' "$*" >&2
    exit 1
}

prefix="$HOME/.local"
while (($#)); do
    case "$1" in
        --prefix)
            if (($# < 2)) || [[ -z $2 || $2 == -* ]]; then
                die "--prefix requires a path"
            fi
            prefix=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (see --help)"
            ;;
    esac
done

[[ $(uname -s) == Linux ]] || die "this installer requires Linux"
command -v python3 >/dev/null || die "python3 is required"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "Python 3.10 or newer is required"
command -v flock >/dev/null || die "flock is required (usually provided by util-linux)"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

# List release files explicitly so calculations and locally prepared inputs are excluded.
required=(
    README.md
    materials-pipeline
    pipeline.py
    terminal.py
    ferro_adapter.py
    mechanical_adapter.py
    optical_adapter.py
    examples/smoke.json
    examples/structures/Si.POSCAR
    examples/structures/BaTiO3.reference.POSCAR
    examples/structures/BaTiO3.polar.POSCAR
    vendor/MechanicalProperties/elastic_workflow.py
    vendor/MechanicalProperties/mech.py
    vendor/MechanicalProperties/mechanical_fit.py
    vendor/MechanicalProperties/mechanical_inputs.py
    vendor/MechanicalProperties/mechanical_vasp_output.py
    vendor/OpticalProperties/optics.py
    vendor/OpticalProperties/optical_workflow/__init__.py
    vendor/OpticalProperties/optical_workflow/__main__.py
    vendor/OpticalProperties/optical_workflow/inputs.py
    vendor/OpticalProperties/optical_workflow/results.py
    vendor/ferroelectricity-dft/scripts/fe_analysis.py
    vendor/ferroelectricity-dft/scripts/fe_structures.py
    vendor/ferroelectricity-dft/scripts/fe_vasp_jobs.py
    vendor/ferroelectricity-dft/scripts/fe_workflow.py
    vendor/ferroelectricity-dft/scripts/polarization_path.py
    vendor/ferroelectricity-dft/scripts/vasp_collect_batio3.py
)
optional=(
    LICENSE
    NOTICE
    requirements.txt
    examples/structures/BaTiO3.source.md
)
for file in "${required[@]}"; do
    [[ -f "$source_dir/$file" && ! -L "$source_dir/$file" ]] || die "missing release file: $file"
done

mkdir -p -- "$prefix"
prefix=$(cd -- "$prefix" && pwd -P)
bin_dir="$prefix/bin"
share_dir="$prefix/share"
releases="$share_dir/.materials-pipeline-releases"
app="$share_dir/materials-pipeline"
commands=(materials-pipeline materials-pipline)
mkdir -p -- "$bin_dir" "$share_dir"
[[ ! -L $releases ]] || die "release directory is a symlink: $releases"
mkdir -p -- "$releases"
exec {lock_fd}>"$releases/.install.lock"
flock "$lock_fd"

if [[ -e $app || -L $app ]]; then
    [[ -L $app ]] || die "existing path is not managed by this installer: $app"
    current=$(readlink -f -- "$app" || true)
    [[ $current == "$releases"/release.* && -d $current &&
        $(dirname -- "$current") == "$releases" ]] ||
        die "existing path is not managed by this installer: $app"
fi
for command in "${commands[@]}"; do
    target="$bin_dir/$command"
    if [[ -e $target || -L $target ]]; then
        [[ -L $target && $(readlink -- "$target") == "$app/materials-pipeline" ]] ||
            die "refusing to replace existing command: $target"
    fi
done

release=$(mktemp -d "$releases/release.XXXXXXXX")
pending_link=""
cleanup() {
    if [[ -n $pending_link ]]; then
        [[ ! -L "$pending_link/app" ]] || rm -- "$pending_link/app"
        rmdir -- "$pending_link"
    fi
    if [[ -n $release && $release == "$releases"/release.* &&
        $(readlink -- "$app" 2>/dev/null || true) != "$release" ]]; then
        rm -rf -- "$release"
    fi
}
trap cleanup EXIT

for file in "${required[@]}" "${optional[@]}"; do
    if [[ -f "$source_dir/$file" && ! -L "$source_dir/$file" ]]; then
        install -Dm 0644 -- "$source_dir/$file" "$release/$file"
    fi
done
chmod 0755 -- "$release" "$release/materials-pipeline"

pending_link=$(mktemp -d "$share_dir/.materials-pipeline-link.XXXXXXXX")
ln -s -- "$release" "$pending_link/app"
mv -Tf -- "$pending_link/app" "$app"
release=""

for command in "${commands[@]}"; do
    target="$bin_dir/$command"
    [[ -L $target ]] || ln -s -- "$app/materials-pipeline" "$target"
done

printf 'Installed Materials Pipeline in %s\n' "$app"
case ":$PATH:" in
    *":$bin_dir:"*) printf 'Start the menu with: materials-pipeline\n' ;;
    *) printf 'Start the menu with: %q\nAdd %q to PATH to use the short command.\n' "$bin_dir/materials-pipeline" "$bin_dir" ;;
esac
