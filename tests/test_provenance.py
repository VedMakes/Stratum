from stratum.provenance import artifact_record, environment_snapshot, verify_run_artifact


def _run(path):
    return {
        "artifact_path": str(path),
        "provenance": {
            "schema_version": 1,
            "artifact": artifact_record(path),
            "environment": environment_snapshot(),
        },
    }


def test_artifact_integrity_verification_accepts_unchanged_local_file(tmp_path):
    artifact = tmp_path / "trusted.joblib"
    artifact.write_bytes(b"local artifact")

    result = verify_run_artifact(_run(artifact))

    assert "error" not in result
    assert result["warnings"] == []


def test_artifact_integrity_verification_rejects_modified_or_legacy_artifacts(tmp_path):
    artifact = tmp_path / "modified.joblib"
    artifact.write_bytes(b"original")
    run = _run(artifact)
    artifact.write_bytes(b"modified")

    modified = verify_run_artifact(run)
    legacy = verify_run_artifact({"artifact_path": str(artifact)})

    assert "checksum mismatch" in modified["error"]
    assert "no verifiable provenance" in legacy["error"]


def test_artifact_integrity_verification_rejects_path_substitution(tmp_path):
    original = tmp_path / "original.joblib"
    substituted = tmp_path / "substituted.joblib"
    original.write_bytes(b"same content")
    substituted.write_bytes(b"same content")
    run = _run(original)
    run["artifact_path"] = str(substituted)

    result = verify_run_artifact(run)

    assert "path differs" in result["error"]


def test_artifact_integrity_verification_warns_on_environment_difference(tmp_path):
    import stratum.provenance as provenance

    artifact = tmp_path / "compatible.joblib"
    artifact.write_bytes(b"artifact")
    run = _run(artifact)
    saved_environment = run["provenance"]["environment"]
    saved_environment["packages"]["scikit_learn"] = "0.0-test"

    result = provenance.verify_run_artifact(run)

    assert "error" not in result
    assert any("scikit_learn version differs" in warning for warning in result["warnings"])
