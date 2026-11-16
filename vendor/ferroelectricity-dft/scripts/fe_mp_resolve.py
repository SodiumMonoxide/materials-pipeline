"""Resolve Materials Project phases and stage the selected fixed-cell structures."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def write_text(path, content):
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as handle:
            pending = Path(handle.name)
            handle.write(content)
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def selection_key(row):
    pair = f"{row['polar_id']}:{row['reference_id']}"
    structure = row["polar_poscar"] + "\0" + row["reference_poscar"]
    return f"{pair}:{hashlib.sha256(structure.encode('utf-8')).hexdigest()[:10]}"


def stage_candidates(directory, rows):
    candidates = []
    for row in rows:
        fields = {key: value for key, value in row.items() if key not in ("polar_poscar", "reference_poscar")}
        if row["status"] == "accepted":
            key = selection_key(row)
            dirname = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
            dest = directory / "candidates" / dirname
            dest.mkdir(parents=True, exist_ok=True)
            for role in ("polar", "reference"):
                filename = dest / f"{role}.POSCAR"
                write_text(filename, row[f"{role}_poscar"])
                fields[f"{role}_poscar"] = filename.relative_to(directory).as_posix()
            fields["selection"] = key
        candidates.append(fields)
    return candidates


def show_options(candidates):
    for index, row in enumerate(candidates, 1):
        strain = row.get("max_abs_strain")
        shift = row.get("rms_displacement_angstrom")
        details = []
        if strain is not None:
            details.append(f"strain {100 * strain:.2f}%")
        if shift is not None:
            details.append(f"RMS {shift:.3f} Å")
        if row.get("reference_matched_spacegroup"):
            details.append(f"aligned {row['reference_matched_spacegroup']}")
        if row.get("reference_energy_above_hull") is not None:
            details.append(f"reference E_hull {row['reference_energy_above_hull']:.3f} eV/atom")
        if row.get("source") == "generated":
            details.append("generated reference")
        detail = f" ({', '.join(details)})" if details else ""
        print(f"  {index}. {row['polar_id']} [{row.get('polar_spacegroup')}] -> "
              f"{row['reference_id']} [{row.get('reference_spacegroup')}]{detail}", flush=True)
        print(f"     --select {row['selection']}", flush=True)


def generated_rows(phase, symprec, max_displacement):
    from fe_mp_geometry import crystal_symmetry, poscar_text
    from fe_mp_parent import synthesize_parent

    polar = phase["structure"]
    identifier = phase["record"]["material_id"]
    polar_text = poscar_text(polar, f"{identifier} polar endpoint")
    proposals = synthesize_parent(polar, symprec=symprec,
                                  max_displacement_angstrom=min(0.3, max_displacement))
    rows = []
    for index, proposal in enumerate(proposals["candidates"], 1):
        reference = proposal["structure"]
        reference_text = poscar_text(reference, f"{identifier} projected reference {index}")
        group = crystal_symmetry(reference, symprec)
        rows.append({
            "status": "accepted", "reason": "symmetrized parent requires confirmation",
            "source": "generated", "polar_id": identifier,
            "reference_id": f"generated-{index}", "polar_spacegroup": phase["spacegroup"],
            "reference_spacegroup": group.international,
            "polar_poscar": polar_text, "reference_poscar": reference_text,
            "reference_centrosymmetric": True, "max_abs_strain": 0,
            "rms_displacement_angstrom": proposal.get("rms_displacement_angstrom"),
            "max_displacement_angstrom": proposal.get("max_displacement_angstrom"),
            "projection": {key: value for key, value in proposal.items() if key != "structure"},
            "requires_human_confirmation": True,
        })
    return rows, proposals["rejections"]


def resolve(directory, cache_dir, *, query, by_id, symprec, max_strain, max_displacement,
            polar_id=None, reference_id=None, selector=None, choose=False, refresh=False):
    from fe_mp_api import fetch_records, material_id_key
    try:
        from fe_mp_geometry import classify, discover
    except ModuleNotFoundError as error:
        if error.name in {"numpy", "spglib", "scipy"}:
            raise ValueError("MP structure preparation needs NumPy and spglib. "
                             "Use the configured WSL Python environment or install the missing packages") from error
        raise

    def canonical_id(value, *, generated=False):
        if value is None or (generated and value.startswith("generated-")):
            return value
        key = material_id_key(value)
        return next((record["material_id"] for record in records
                     if material_id_key(record["material_id"]) == key), value)

    records = fetch_records(query, by_id=by_id, cache_dir=cache_dir, refresh=refresh)
    polar_id = canonical_id(polar_id)
    reference_id = canonical_id(reference_id, generated=True)
    phases = classify(records, symprec=symprec)
    resolved_record = next((phase for phase in phases if phase["record"].get("requested_id")), None)
    if by_id and resolved_record is None:
        raise ValueError(f"{query}: the MP response did not identify the requested structure")
    blocked = None
    if resolved_record is not None:
        canonical = resolved_record["record"]["material_id"]
        if resolved_record["classification"] == "polar":
            if polar_id and polar_id not in (canonical, query):
                raise ValueError(f"{query} pins polar structure {canonical}")
            polar_id = canonical
        elif resolved_record["classification"] == "centrosymmetric":
            if reference_id and reference_id not in (canonical, query):
                raise ValueError(f"{query} pins reference structure {canonical}")
            reference_id = canonical
        else:
            blocked = f"{canonical}: {resolved_record['reason']}"
    rows = discover(records, symprec=symprec, max_strain=max_strain,
                    max_displacement=max_displacement)
    for row in rows:
        row.setdefault("source", "materials_project")

    def eligible(row):
        return (blocked is None and row["status"] == "accepted"
                and (not polar_id or row["polar_id"] == polar_id)
                and (not reference_id or row["reference_id"] == reference_id))

    database_available = any(row["status"] == "accepted"
                             and (not polar_id or row["polar_id"] == polar_id) for row in rows)
    projected_rejections = {}
    if (blocked is None and (not reference_id or reference_id.startswith("generated-"))
            and not database_available):
        for phase in phases:
            if phase["classification"] == "polar" and (
                    not polar_id or phase["record"]["material_id"] == polar_id):
                projected, rejections = generated_rows(phase, symprec, max_displacement)
                rows.extend(projected)
                if rejections:
                    projected_rejections[phase["record"]["material_id"]] = rejections

    directory.mkdir(parents=True, exist_ok=True)
    candidates = stage_candidates(directory, rows)
    viable = [row for row in candidates if eligible(row)]
    for row in candidates:
        row["matches_query"] = eligible(row)
    report_path = directory / "candidates.json"
    report = {
        "query": query, "query_type": "material_id" if by_id else "formula",
        "resolved_material_id": (resolved_record["record"]["material_id"] if resolved_record else None),
        "filters": {"polar_id": polar_id, "reference_id": reference_id},
        "thresholds": {"symprec_angstrom": symprec, "max_strain": max_strain,
                       "max_displacement_angstrom": max_displacement},
        "phases": [{**{key: phase["record"].get(key) for key in
                       ("material_id", "formula_pretty", "band_gap", "energy_above_hull", "last_updated")},
                    **{key: phase.get(key) for key in
                       ("classification", "reason", "spacegroup", "point_group")}} for phase in phases],
        "candidates": candidates,
        "projected_reference_rejections": projected_rejections,
    }
    write_json(report_path, report)

    remembered_path = directory / "selection.json"
    selected = None
    explicit = False
    stale_selection = False
    if selector is not None:
        selected = next((row for row in viable if row["selection"] == selector), None)
        explicit = True
        if selected is None:
            raise ValueError(f"selected pair is unavailable for this query. See {report_path}")
    elif remembered_path.is_file():
        prior = json.loads(remembered_path.read_text(encoding="utf-8"))
        if (prior.get("query") == query and prior.get("query_type") == report["query_type"]
                and prior.get("explicit") is True):
            selected = next((row for row in viable if row["selection"] == prior.get("selection")), None)
            explicit = selected is not None
            stale_selection = not explicit

    if selected is None and not stale_selection and len(viable) == 1:
        only = viable[0]
        if only["source"] != "generated" and not only.get("alignment_ambiguous"):
            selected = only
    if selected is None and viable:
        show_options(viable)
        if choose and sys.stdin.isatty():
            try:
                reply = input("Choose a pair by number (Enter to defer): ").strip()
            except EOFError:
                reply = ""
            if reply:
                if not reply.isascii() or not reply.isdecimal() or not 1 <= int(reply) <= len(viable):
                    raise ValueError(f"expected a number from 1 to {len(viable)}")
                selected, explicit = viable[int(reply) - 1], True

    if selected is None:
        status = "needs_selection" if viable else "no_compatible_structure"
        message = (f"{len(viable)} candidate pair(s) need your selection"
                   if viable else blocked or "No compatible polar structure and nonpolar reference")
        if not viable and database_available and reference_id and reference_id.startswith("generated-"):
            message = "A compatible database reference is available. Select a Materials Project pair"
        if stale_selection:
            message = f"Previously confirmed pair is unavailable. {message}"
        return {"status": status, "message": f"{message}. See {report_path}",
                "candidate_report": str(report_path), "candidate_count": len(viable)}
    if selected.get("alignment_ambiguous") and not selected.get("alignment_id"):
        return {"status": "needs_selection", "candidate_report": str(report_path),
                "candidate_count": len(viable),
                "message": f"Ambiguous cell alignment. Inspect {report_path} and supply aligned POSCARs"}

    source = {
        "query": query, "query_type": report["query_type"], "selection": selected["selection"],
        "source": selected["source"], "polar_id": selected["polar_id"],
        "reference_id": selected["reference_id"], "explicit": explicit,
    }
    write_json(remembered_path, source)
    return {"status": "resolved", "candidate_report": str(report_path),
            "reference": directory / selected["reference_poscar"],
            "polar": directory / selected["polar_poscar"],
            "mirror": bool(selected.get("reference_centrosymmetric", True)),
            "structure_source": {key: value for key, value in source.items() if key != "explicit"}}
