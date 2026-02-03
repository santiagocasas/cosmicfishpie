from importlib import resources

import yaml


def test_packaged_config_files_exist_and_parse():
    cfg_dir = resources.files("cosmicfishpie") / "configs"

    yaml_paths = [
        cfg_dir / "default_survey_specifications" / "Euclid-Photometric-ISTF-Pessimistic.yaml",
        cfg_dir / "default_survey_specifications" / "Euclid-Spectroscopic-ISTF-Pessimistic.yaml",
        cfg_dir / "default_boltzmann_yaml_files" / "camb" / "default.yaml",
        cfg_dir / "default_boltzmann_yaml_files" / "class" / "default.yaml",
        cfg_dir / "default_boltzmann_yaml_files" / "symbolic" / "default.yaml",
    ]

    for path in yaml_paths:
        assert path.is_file(), f"missing packaged config file: {path}"
        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert parsed is not None

    dat_path = cfg_dir / "external_data" / "lumratio_file.dat"
    assert dat_path.is_file(), f"missing packaged data file: {dat_path}"
