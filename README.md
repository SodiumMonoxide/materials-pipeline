# Materials Pipeline

Materials Pipeline runs VASP workflows for ferroelectric polarization, electronic optical spectra, and bulk elastic properties. A JSON manifest selects the calculations for each material. Use the numbered terminal menu for interactive work or the command line for scripts and repeatable batches.

## Install on Linux

You need Python 3.10 or later, Bash, GNU coreutils, util-linux (`taskset` and `flock`), and an Open MPI `mpirun` compatible with your VASP build. Supply your licensed VASP executable and PAW-PBE datasets. The project contains workflow code and example structures.

From a checkout or extracted source archive, create a Python environment and install the launcher:

```bash
python3 -m venv "$HOME/.venvs/materials-pipeline"
source "$HOME/.venvs/materials-pipeline/bin/activate"
python -m pip install -r requirements.txt
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
```

The default install places commands in `$HOME/.local/bin` and program files in `$HOME/.local/share/materials-pipeline`. For another location, use `bash install.sh --prefix /absolute/path` and adapt the paths below to that prefix. To upgrade, run the installer again from the new source using the same prefix. Activate the Python environment in each shell used to run the pipeline, or set `runtime.python` in the manifest to the interpreter in that environment.

Set the PAW-PBE library path. The bundled example expects datasets such as `Si/POTCAR`, `Ba_sv/POTCAR`, `Ti_sv/POTCAR`, and `O/POTCAR` under this directory.

```bash
export VASP_POTCAR_ROOT=/absolute/path/to/PAW-PBE
# Set this if vasp_std is outside PATH.
export VASP_EXECUTABLE=/absolute/path/to/vasp_std
```

## Try the example

```bash
sample="$HOME/.local/share/materials-pipeline/examples/smoke.json"
output="$PWD/materials-smoke"
materials-pipeline doctor "$sample"
materials-pipeline prepare "$sample" --out "$output"
materials-pipeline run "$sample" --out "$output"
```

`doctor` checks the environment. `prepare` generates or previews inputs without launching VASP. `run` starts VASP and writes results. The bundled diamond silicon and tetragonal BaTiO₃ structures are idealized geometries intended for a workflow smoke test.

Launch `materials-pipeline` without arguments to open the terminal menu. The `materials-pipline` alias opens the same menu. Choose 1 through 3 for polarization, optics, or mechanics, or choose 4 for the full manifest. The submenu offers preparation and execution. Choose 11 for the environment check, 12 for the latest batch summary, 13 for ferroelectric structure discovery, or 0 to exit. Enter accepts the displayed defaults, including `./runs/<manifest-stem>` for the output directory.

## Define a batch

Copy the entire `examples/` directory to a working directory if you want to edit the example and keep its relative structure paths. Relative paths for structures and inputs resolve from the manifest's directory. A relative `--out` path resolves from the current directory.

A manifest contains `materials`, plus optional `runtime` and `defaults` objects. IDs must be unique and use ASCII letters, digits, periods, underscores, or hyphens. Start each ID with a letter or digit. Set a property to `true` to use its defaults, `false` to disable it, or an object to override settings for that material.

```json
{
  "runtime": {"workers": 2, "ranks": 2, "threads": 1},
  "defaults": {
    "mechanical": {"preset": "standard"},
    "optical": {"profile": "standard", "omegamax": 12.0}
  },
  "materials": [
    {
      "id": "silicon",
      "structure": "structures/Si.POSCAR",
      "properties": {
        "mechanical": true,
        "optical": {"structure_from": "mechanical", "component": "tensor"}
      }
    }
  ]
}
```

To set machine-specific paths, use `runtime.vasp`, `runtime.mpi`, `runtime.python`, and `runtime.paw_dir`. These accept explicit paths. For executables you can also give a command in `PATH`. By default, the pipeline uses the current Python interpreter, `VASP_EXECUTABLE` or `vasp_std`, `mpirun`, and the `VASP_POTCAR_ROOT` directory. Manifest path values support `~` and environment variables.

| Property | Inputs and method | Main outputs |
| --- | --- | --- |
| `ferroelectric` | Supply an insulating reference POSCAR via `reference` and a polar endpoint via `structure` or `polar`. Use the same fixed cell for both. Supply `opposite` or set `mirror: true` for a two-sided path. The workflow checks the gap and unwraps the Berry-phase polarization branch. | `summary.json`, `path.csv`, and `unwrapped.csv`. Polarization differences in C/m² and energy in eV per input cell. |
| `optical` | Supply a POSCAR for a fixed-geometry `LOPTICS` calculation. Set `"structure_from": "mechanical"` to use the equilibrium structure from a successful mechanical stage for the same material. | `spectrum.csv` and `run.json`. Energy in eV, a dielectric spectrum, and optical constants when requested and applicable. |
| `mechanical` | Supply a POSCAR, CIF, or another single-structure format supported by ASE. The workflow checks the equilibrium cell and computes relaxed-ion elasticity using `method: "auto"`, `"vasp"`, or `"stress-strain"`. | `result.json` with a 6×6 stiffness tensor, stability checks, and available Voigt–Reuss–Hill moduli. Stresses and moduli in GPa. |

Ferroelectric and optical POSCAR files should include element names in VASP 5/6 format. The polarization path needs a suitable nonpolar reference and consistent atom mapping. The sampled energy maximum describes the chosen path. The optical calculation uses the independent-particle electronic response, and the elastic workflow targets three-dimensional ordered bulk systems near zero applied pressure. For research runs, check convergence against k meshes, cutoff energies, optical empty bands, polarization path sampling, and strain amplitudes as applicable.

## Discover ferroelectric structures

A `ferroelectric` stage can use `formula` or `material_id` to select a polar endpoint and a nonpolar reference from Materials Project. Either query replaces the local `reference`, `polar`, and `opposite` inputs. A top-level `structure` is optional when every enabled stage can obtain its structure from the query or an upstream stage. The bundled `examples/discovery.json` shows a formula search with an optical follow-up.

```bash
read -rsp 'Materials Project API key: ' MP_API_KEY
printf '\n'
export MP_API_KEY
manifest="$HOME/.local/share/materials-pipeline/examples/discovery.json"
materials-pipeline discover "$manifest" --out runs/batio3
```

`discover` writes candidate pairs and aligned POSCARs without starting VASP. If the status is `needs_selection`, open the candidate report listed in the summary and copy the complete `selection` value for a suitable pair into the material's `ferroelectric` settings. You can also narrow the search using `polar_id` and `reference_id`. Review the structural match before submitting a calculation. Use the same output directory for later `prepare` or `run` calls to keep the downloaded records available. A custom `mp_cache` can point to a cache directory or a local Materials Project response JSON. Use `refresh_mp: true` to refresh a directory cache when appropriate.

Set `"structure_from": "ferroelectric"` in `optical` or `mechanical` to use the selected polar endpoint. In preparation mode the endpoint is available once the ferroelectric inputs are prepared. In run mode the pipeline requires a completed ferroelectric stage before starting the dependent stage. Explicit optical `potentials` or mechanical `potcars` entries can override inherited PAW variant names. Selecting a dependent stage also selects its prerequisites.

## Check convergence

Optical and elastic convergence are optional. Ordinary screening can use the original single-calculation settings.

| Stage | Manifest options | Comparison and report |
| --- | --- | --- |
| `optical` | Set `convergence: true` for defaults, or supply an object. Available keys are `window_eV` (default `[0, 8]`), `rtol` (0.05), `atol` (0.02), `k_growth` (1.4), `k_steps` (5), `kmeshes` (explicit finer meshes), `band_growth` (1.4), and `band_steps` (3). | Compare the dielectric spectra while independently refining the k mesh and total band count. The report includes component errors, the tested spectra, and a plot. |
| `mechanical` | Set `converge: true` to enable the checks. Set `convergence_tol_gpa`, `convergence_max_rounds`, `convergence_k_step`, and `convergence_encut_step` to override the defaults of 1 GPa, 3 rounds, 2 mesh points per axis, and 80 eV. | Check k-mesh and cutoff changes independently against the maximum absolute difference among the 36 stiffness entries. The JSON report records the tested settings and tensor differences. |

The summary records the convergence status and links to the report and final available spectrum or elastic tensor. A stage marked `not_converged` exhausted its configured budget. Repeat with the same inputs and output directory to reuse matching jobs when adjusting tolerances or budgets.

## Run and inspect a batch

```bash
materials-pipeline run batch.json --out results/batch-01 --workers 2
materials-pipeline run batch.json --out results/batch-01 --material silicon --property optical
```

- Both selection flags can be repeated. Selecting a stage with `structure_from` also selects the prerequisites it needs. A stage that requires a relaxed mechanical structure is marked `deferred` during preparation.
- The pipeline runs at most one stage at a time for each material. Allow `workers × ranks × threads` logical CPUs for the active workers and account for memory and scratch space separately. The optical workflow uses one thread per rank.
- The output directory contains CSV and JSON summaries named for the action, such as `discover-summary.json`, `prepare-summary.json`, or `run-summary.json`. Each row links to the stage log, result, and output directory when available. Open `request.json` in the stage directory for the input hashes and runtime settings.
- A `completed` stage passed the workflow checks. A `review` stage produced a result that needs inspection. `blocked` indicates an unavailable prerequisite or an unmet physics check, and `failed` indicates an error. Inspect `driver.log` and the relevant VASP logs. Independent jobs continue when a stage fails.
- To resume or reuse completed work, repeat the command with the same inputs and output directory. Changed inputs or settings produce a separate stage directory. Use a different output root for each simultaneous batch.

CLI exit codes are 0 for completion, 1 for a failed, blocked, or not-converged stage, 2 for an invalid environment or manifest, 3 when the outstanding issue is a candidate selection, and 130 for interruption. Inspect the summary even on exit 0 to find any `review` results.

Mechanical input preparation writes licensed `POTCAR` data into job directories. Keep the output tree local. To share results, select only the intended files and review them for local paths and licensed data.

## 中文快速开始

在 Linux 或 WSL 的终端中，按上文安装 Python 依赖并运行 `bash install.sh`。设置 `VASP_POTCAR_ROOT`，将 VASP 加入 PATH 或设置 `VASP_EXECUTABLE`。激活安装依赖时使用的 Python 环境后，输入 `materials-pipeline` 打开菜单。选择 1、2、3 分别进入铁电、光学、力学计算，选择 4 运行清单。可先使用 `doctor` 和 `prepare` 检查环境与输入。
