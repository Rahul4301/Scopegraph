import pytest

import memstudy.preflight as pf


def test_paid_commands_need_the_prereg_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "prereg_committed", lambda tag=pf.PREREG_TAG: False)
    path = tmp_path / "a.yaml"
    path.write_text("stage_pilot: true\n")
    with pytest.raises(pf.NotApproved, match="prereg-v1"):
        pf.require_gates("stage_pilot", approvals_path=path)


def test_gate_must_be_true(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "prereg_committed", lambda tag=pf.PREREG_TAG: True)
    path = tmp_path / "a.yaml"
    path.write_text("stage_pilot: false\ng1_extra_models: true\n")
    with pytest.raises(pf.NotApproved, match="stage_pilot"):
        pf.require_gates("stage_pilot", approvals_path=path)
    pf.require_gates("g1_extra_models", approvals_path=path)


def test_shipped_approvals_are_all_closed():
    import yaml

    gates = yaml.safe_load(pf.APPROVALS.read_text())
    assert gates and not any(gates.values())


def test_key_loader_reads_only_the_openai_key_and_never_echoes_it(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=other\nOPENAI_API_KEY=\"sk-secret\"\n")
    pf.load_openai_key(env)
    assert capsys.readouterr().out == ""
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-secret"


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(pf.NotApproved):
        pf.load_openai_key(tmp_path / "missing.env")


def test_key_var_lets_a_differently_named_variable_supply_the_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=sk-other\nOPENAI_API_KEY=\n")
    pf.load_openai_key(env, key_var="LLM_API_KEY")
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-other" and capsys.readouterr().out == ""
