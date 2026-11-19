import pytest
from mocks import ExecResult, FakeContainer, FakeDocker, FakeSandboxManager

from ideaforge.errors import CommandBlockedError, ExecTimeoutError, SandboxError
from ideaforge.sandbox.manager import ContainerSpec, SandboxManager, safe_image_name


def test_host_network_spec_rejected():
    with pytest.raises(SandboxError):
        ContainerSpec(image="python:3.11-slim", name="c", network="host")


def test_bridge_spec_accepted():
    spec = ContainerSpec(image="python:3.11-slim", name="c", network="bridge")
    assert spec.mem_limit == "4g"
    assert spec.cpu_count == 2.0


def test_resource_limits_passed_to_docker():
    docker = FakeDocker()
    manager = SandboxManager(docker_client=docker)
    spec = ContainerSpec(image="python:3.11-slim", name="c", network="bridge",
                         mem_limit="4g", cpu_count=2.0)
    manager.create(spec)
    created = docker.created[0]
    assert created.mem_limit == "4g"
    assert created.cpu_count == 2.0
    assert created.network == "bridge"


def test_exec_success_and_failure():
    sandbox = FakeSandboxManager(exec_script=[
        ExecResult(success=True, exit_code=0, output="all good"),
        ExecResult(success=False, exit_code=1, output="boom"),
    ])
    handle = FakeContainer()
    ok = sandbox.exec(handle, ["pytest", "-q"])
    assert ok.success and ok.output == "all good"
    bad = sandbox.exec(handle, ["pytest", "-q"])
    assert not bad.success and bad.output == "boom"


def test_exec_timeout_retries_then_raises():
    from ideaforge.errors import ExecTimeoutError

    timeout = ExecTimeoutError("timed out")
    sandbox = FakeSandboxManager(exec_script=[timeout, timeout, timeout])
    handle = FakeContainer()
    with pytest.raises(ExecTimeoutError):
        sandbox.exec(handle, ["pytest"], retry=2)
    assert len(sandbox.exec_calls) == 3  # 1 + 2 retries


def test_exec_timeout_recovers_within_retry():
    timeout = ExecTimeoutError("timed out once")
    sandbox = FakeSandboxManager(exec_script=[
        timeout,
        ExecResult(success=True, exit_code=0, output="recovered"),
    ])
    handle = FakeContainer()
    result = sandbox.exec(handle, ["pytest"], retry=2)
    assert result.success


def test_blocked_command_raises_and_journals():
    journal_calls = []
    sandbox = FakeSandboxManager()
    sandbox.journal = type("J", (), {"emit": lambda self, t, p, cycle=0: journal_calls.append((t, p))})()
    handle = FakeContainer()
    with pytest.raises(CommandBlockedError):
        sandbox.exec(handle, ["bash", "-lc", "rm -rf /"])
    assert journal_calls and journal_calls[0][0] == "command_blocked"


def test_image_naming_and_cache():

    sandbox = FakeSandboxManager(images={"donor_repo"})
    assert sandbox.ensure_image("donor/repo") is not None
    assert sandbox.ensure_image("missing/repo") is None
    assert safe_image_name("donor/repo") == "donor_repo"


def test_commit_image_tagged():
    sandbox = FakeSandboxManager()
    ref = sandbox.commit_image(FakeContainer(), "donor_repo")
    assert ref.tag == "donor_repo:latest"
    assert sandbox.committed == ["donor_repo"]


def test_destroy_counts():
    sandbox = FakeSandboxManager()
    handle = FakeContainer()
    sandbox.destroy(handle)
    sandbox.destroy(handle)
    assert sandbox.destroyed == 2