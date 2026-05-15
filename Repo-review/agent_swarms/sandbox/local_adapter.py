"""Workspace-constrained local sandbox backend used for API verification."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import sys
import sysconfig
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from agent_swarms.sandbox.base import (
    ExecutionRecord,
    SandboxAdapter,
    SandboxExecutionError,
    SandboxFileRecord,
    SandboxSession,
)
from agent_swarms.sandbox.path_utils import resolve_within_root
from agent_swarms.sandbox.workspace_manager import WorkspaceManager
from agent_swarms.state.enums import SandboxBackend, SandboxRunStatus

NETWORK_COMMANDS = frozenset(
    {
        "curl",
        "dig",
        "ftp",
        "host",
        "nc",
        "ncat",
        "ping",
        "scp",
        "sftp",
        "ssh",
        "telnet",
        "traceroute",
        "wget",
    }
)
COMMAND_PREFIXES = frozenset({"builtin", "command", "env", "nice", "nohup", "time"})
COMMAND_SEPARATORS = frozenset({"&&", ";", "|", "||"})
REDIRECTION_TOKENS = frozenset({"<", ">", ">>"})
SPECIAL_ALLOWED_PATHS = frozenset(
    {
        Path("/dev/null"),
        Path("/dev/tty"),
        Path("/private/dev/null"),
        Path("/private/dev/tty"),
    }
)
DEFAULT_MACOS_READ_ROOTS = (
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/System"),
    Path("/Library"),
    Path("/opt/homebrew"),
)
MACOS_DENIED_READ_PATHS = (
    Path("/private/etc/group"),
    Path("/private/etc/hosts"),
    Path("/private/etc/master.passwd"),
    Path("/private/etc/passwd"),
    Path("/private/etc/protocols"),
    Path("/private/etc/resolv.conf"),
    Path("/private/etc/services"),
    Path("/private/etc/ssl/cert.pem"),
    Path("/private/etc/ssl/openssl.cnf"),
    Path("/private/var/run/resolv.conf"),
)
FORBIDDEN_SHELL_PATTERNS = (
    (
        re.compile(r"`"),
        "Command substitution with backticks is not allowed in the local sandbox.",
    ),
    (
        re.compile(r"\$\("),
        "Command substitution is not allowed in the local sandbox.",
    ),
    (
        re.compile(r"<<<?"),
        "Here-doc shell redirection is not allowed in the local sandbox.",
    ),
)
PYTHON_SANDBOX_PREAMBLE = textwrap.dedent(
    """
    import asyncio
    import builtins
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import sysconfig

    _WORKSPACE_ROOT = Path(__file__).resolve().parent
    _ALLOWED_SPECIAL_PATHS = {
        Path("/dev/null"),
        Path("/dev/tty"),
        Path("/private/dev/null"),
        Path("/private/dev/tty"),
    }
    _READ_ALLOWED_ROOTS = {_WORKSPACE_ROOT}

    for _key in ("platlib", "platstdlib", "purelib", "scripts", "stdlib"):
        _resolved_path = sysconfig.get_path(_key)
        if _resolved_path:
            _READ_ALLOWED_ROOTS.add(Path(_resolved_path).resolve())

    for _candidate_path in (
        Path(sys.executable).resolve().parent,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.exec_prefix).resolve(),
        Path(sys.base_exec_prefix).resolve(),
    ):
        if _candidate_path.exists():
            _READ_ALLOWED_ROOTS.add(_candidate_path)

    def _is_write_mode(mode):
        return any(flag in mode for flag in ("+", "a", "w", "x"))

    def _is_within_allowed_roots(resolved):
        return any(resolved.is_relative_to(root) for root in _READ_ALLOWED_ROOTS)

    def _resolve_workspace_path(value, *, write=False):
        if isinstance(value, int):
            return value
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str):
            return value

        expanded = os.path.expanduser(value)
        candidate = Path(expanded)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (_WORKSPACE_ROOT / candidate).resolve()

        if resolved in _ALLOWED_SPECIAL_PATHS:
            return value
        if resolved.is_relative_to(_WORKSPACE_ROOT):
            return value
        if not write and _is_within_allowed_roots(resolved):
            return value
        raise PermissionError(f"Path '{value}' escapes the local sandbox workspace.")

    def _guard_path_operation(func, *, write=False):
        def _wrapped(path=".", *args, **kwargs):
            _resolve_workspace_path(path, write=write)
            return func(path, *args, **kwargs)
        return _wrapped

    _original_open = builtins.open

    def _guarded_open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        _resolve_workspace_path(file, write=_is_write_mode(mode))
        return _original_open(file, *args, **kwargs)

    builtins.open = _guarded_open

    _original_os_open = os.open

    def _guarded_os_open(path, *args, **kwargs):
        flags = args[0] if args else kwargs.get("flags", 0)
        write_flags = (
            os.O_APPEND
            | os.O_CREAT
            | os.O_RDWR
            | os.O_TRUNC
            | os.O_WRONLY
        )
        _resolve_workspace_path(path, write=bool(flags & write_flags))
        return _original_os_open(path, *args, **kwargs)

    os.open = _guarded_os_open

    _original_chdir = os.chdir

    def _guarded_chdir(path):
        _resolve_workspace_path(path)
        return _original_chdir(path)

    os.chdir = _guarded_chdir
    os.listdir = _guard_path_operation(os.listdir)
    os.scandir = _guard_path_operation(os.scandir)
    os.walk = _guard_path_operation(os.walk)
    os.remove = _guard_path_operation(os.remove, write=True)
    os.unlink = _guard_path_operation(os.unlink, write=True)
    os.mkdir = _guard_path_operation(os.mkdir, write=True)
    os.makedirs = _guard_path_operation(os.makedirs, write=True)
    os.rmdir = _guard_path_operation(os.rmdir, write=True)
    os.chmod = _guard_path_operation(os.chmod, write=True)

    _original_rename = os.rename

    def _guarded_rename(src, dst, *args, **kwargs):
        _resolve_workspace_path(src)
        _resolve_workspace_path(dst, write=True)
        return _original_rename(src, dst, *args, **kwargs)

    os.rename = _guarded_rename
    os.replace = _guarded_rename

    _original_path_open = Path.open

    def _guarded_path_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        _resolve_workspace_path(self, write=_is_write_mode(mode))
        return _original_path_open(self, *args, **kwargs)

    Path.open = _guarded_path_open

    def _guarded_path_method(method_name, *, write=False):
        original = getattr(Path, method_name)

        def _wrapped(self, *args, **kwargs):
            _resolve_workspace_path(self, write=write)
            return original(self, *args, **kwargs)

        return _wrapped

    for _method_name in ("glob", "iterdir", "read_bytes", "read_text", "rglob"):
        setattr(Path, _method_name, _guarded_path_method(_method_name))

    for _method_name in (
        "chmod",
        "mkdir",
        "rmdir",
        "unlink",
        "write_bytes",
        "write_text",
    ):
        setattr(Path, _method_name, _guarded_path_method(_method_name, write=True))

    def _deny_subprocess(*args, **kwargs):
        raise PermissionError(
            "Subprocess execution is disabled in the local sandbox Python runner."
        )

    subprocess.Popen = _deny_subprocess
    subprocess.run = _deny_subprocess
    subprocess.call = _deny_subprocess
    subprocess.check_call = _deny_subprocess
    subprocess.check_output = _deny_subprocess
    asyncio.create_subprocess_exec = _deny_subprocess
    asyncio.create_subprocess_shell = _deny_subprocess

    def _deny_network(*args, **kwargs):
        raise PermissionError("Network access is disabled in the local sandbox Python runner.")

    socket.socket = _deny_network
    socket.create_connection = _deny_network
    socket.socketpair = _deny_network

    os.chdir(_WORKSPACE_ROOT)
    """
).strip()


class LocalSandboxAdapter(SandboxAdapter):
    """Execute commands inside a per-run local workspace."""

    backend = SandboxBackend.LOCAL

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        *,
        use_macos_profile: bool = True,
    ) -> None:
        self._workspace_manager = workspace_manager
        self._use_macos_profile = use_macos_profile
        self._sandbox_exec_path = shutil.which("sandbox-exec")

    async def create_session(self, run_id: str) -> SandboxSession:
        """Create local sandbox session metadata."""

        return SandboxSession(
            run_id=run_id,
            sandbox_backend=self.backend,
            workspace_path=Path(),
            created_at=datetime.now(UTC),
            status=SandboxRunStatus.READY,
        )

    async def mount_repo(
        self,
        session: SandboxSession,
        repo_source_path: Path | None,
    ) -> SandboxSession:
        """Stage the requested repository into a local workspace."""

        self._workspace_manager.prepare_workspace(session.workspace_path, repo_source_path)
        session.repo_source_path = repo_source_path
        return session

    async def execute_shell(
        self,
        session: SandboxSession,
        command: str,
        timeout_seconds: int,
    ) -> ExecutionRecord:
        """Execute an allowlisted shell command in the local workspace."""

        self._validate_shell_command(command, session.workspace_path)
        return await self._run_process(
            args=("/bin/sh", "-lc", command),
            recorded_command=command,
            cwd=session.workspace_path,
            timeout_seconds=timeout_seconds,
        )

    async def execute_python(
        self,
        session: SandboxSession,
        code: str,
        timeout_seconds: int,
        python_executable: str,
    ) -> ExecutionRecord:
        """Execute guarded Python code in the local workspace."""

        script_path = await self._write_temp_script(
            session.workspace_path,
            self._build_guarded_python_code(code),
        )
        try:
            command = f"{python_executable} -I {script_path.name}"
            return await self._run_process(
                args=(python_executable, "-I", script_path.name),
                recorded_command=command,
                cwd=session.workspace_path,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if script_path.exists():
                script_path.unlink()

    async def list_files(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> list[SandboxFileRecord]:
        """List files inside the local workspace."""

        target_dir = resolve_within_root(session.workspace_path, relative_path)
        if not target_dir.exists():
            raise FileNotFoundError(f"Path does not exist: {relative_path}")
        if not target_dir.is_dir():
            raise ValueError(f"Path is not a directory: {relative_path}")

        return [
            SandboxFileRecord(
                path=str(path.relative_to(session.workspace_path)),
                is_dir=path.is_dir(),
                size_bytes=path.stat().st_size if path.is_file() else 0,
            )
            for path in sorted(target_dir.iterdir(), key=lambda item: item.name)
        ]

    async def read_file(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> str:
        """Read a text file from the local workspace."""

        target_file = resolve_within_root(session.workspace_path, relative_path)
        if not target_file.exists():
            raise FileNotFoundError(f"File does not exist: {relative_path}")
        if not target_file.is_file():
            raise ValueError(f"Path is not a file: {relative_path}")
        return target_file.read_text()

    async def write_file(
        self,
        session: SandboxSession,
        relative_path: str,
        content: str,
    ) -> None:
        """Write a text file into the local workspace."""

        target_file = resolve_within_root(session.workspace_path, relative_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content)

    async def close_session(self, session: SandboxSession) -> SandboxSession:
        """Mark the local workspace session closed."""

        session.status = SandboxRunStatus.CLOSED
        session.closed_at = datetime.now(UTC)
        return session

    async def _run_process(
        self,
        *,
        args: tuple[str, ...],
        recorded_command: str,
        cwd: Path,
        timeout_seconds: int,
    ) -> ExecutionRecord:
        started_at = datetime.now(UTC)
        tmp_dir = cwd / ".sandbox_tmp"
        tmp_dir.mkdir(exist_ok=True)
        env = self._build_environment(cwd=cwd, tmp_dir=tmp_dir)

        process_args = args
        using_macos_profile = self._should_use_macos_profile()
        if using_macos_profile:
            process_args = self._wrap_with_macos_profile(args=args, cwd=cwd)

        stdout_raw, stderr_raw, exit_code = await self._communicate(
            args=process_args,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            recorded_command=recorded_command,
        )

        if using_macos_profile and self._should_fallback_from_profile(stderr_raw, exit_code):
            stdout_raw, stderr_raw, exit_code = await self._communicate(
                args=args,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                recorded_command=recorded_command,
            )

        finished_at = datetime.now(UTC)
        return ExecutionRecord(
            command=recorded_command,
            exit_code=exit_code,
            stdout=stdout_raw.decode("utf-8", errors="replace"),
            stderr=stderr_raw.decode("utf-8", errors="replace"),
            duration_ms=int((finished_at - started_at).total_seconds() * 1000),
            executed_at=started_at,
        )

    async def _communicate(
        self,
        *,
        args: tuple[str, ...],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        recorded_command: str,
    ) -> tuple[bytes, bytes, int]:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise SandboxExecutionError(
                f"Command timed out after {timeout_seconds} seconds: {recorded_command}"
            ) from exc

        return stdout_raw, stderr_raw, process.returncode

    def _validate_shell_command(self, command: str, workspace_path: Path) -> None:
        stripped_command = command.strip()
        if not stripped_command:
            raise ValueError("Shell command cannot be empty.")

        for pattern, message in FORBIDDEN_SHELL_PATTERNS:
            if pattern.search(stripped_command):
                raise ValueError(message)

        try:
            tokens = self._tokenize_shell_command(stripped_command)
        except ValueError as exc:
            raise ValueError(
                "Shell command could not be parsed safely for local execution."
            ) from exc

        expecting_command = True
        for index, token in enumerate(tokens):
            if token in COMMAND_SEPARATORS:
                expecting_command = True
                continue

            if token in REDIRECTION_TOKENS:
                next_token = self._next_token(tokens, index + 1)
                if next_token is None:
                    raise ValueError("Shell redirection is missing a target path.")
                self._validate_path_token(next_token, workspace_path)
                continue

            if expecting_command:
                if self._is_env_assignment(token):
                    self._validate_assignment_value(token, workspace_path)
                    continue

                executable_name = Path(token).name
                if executable_name in COMMAND_PREFIXES:
                    continue
                if executable_name in NETWORK_COMMANDS:
                    raise ValueError(
                        f"Command '{executable_name}' is not allowed in the local sandbox because "
                        "network access is disabled."
                    )
                expecting_command = False

            self._validate_path_token(token, workspace_path)

    def _validate_assignment_value(self, token: str, workspace_path: Path) -> None:
        _, raw_value = token.split("=", maxsplit=1)
        if not raw_value:
            return

        for part in raw_value.split(os.pathsep):
            if self._looks_like_path(part):
                self._validate_path_token(part, workspace_path)

    def _validate_path_token(self, token: str, workspace_path: Path) -> None:
        if not self._looks_like_path(token):
            return

        target = token
        if target.startswith("~/"):
            target = target.replace("~", ".", 1)
        elif target == "~":
            target = "."

        if target.startswith("/"):
            resolved_target = Path(target).resolve()
            if resolved_target in SPECIAL_ALLOWED_PATHS:
                return
            workspace_root = workspace_path.resolve()
            if resolved_target.is_relative_to(workspace_root):
                return
            raise ValueError(f"Path '{token}' is outside the sandbox workspace.")

        resolve_within_root(workspace_path, target)

    def _should_use_macos_profile(self) -> bool:
        return bool(
            self._use_macos_profile
            and self._sandbox_exec_path is not None
            and sys.platform == "darwin"
        )

    def _wrap_with_macos_profile(self, *, args: tuple[str, ...], cwd: Path) -> tuple[str, ...]:
        return (
            self._sandbox_exec_path or "sandbox-exec",
            "-D",
            f"WORKSPACE={cwd.resolve()}",
            "-p",
            self._build_macos_profile(cwd),
            *args,
        )

    def _build_macos_profile(self, cwd: Path) -> str:
        runtime_roots = self._collect_runtime_read_roots(cwd)
        runtime_rules = "\n".join(self._sbpl_subpath_rule(path) for path in runtime_roots)
        denied_rules = "\n".join(
            self._sbpl_literal_deny_rule(path) for path in MACOS_DENIED_READ_PATHS
        )

        return textwrap.dedent(
            f"""\
            (version 2)
            (deny default)

            (import "system.sb")
            (import "com.apple.corefoundation.sb")

            (deny network*)
            {denied_rules}

            (allow syscall*)
            (allow system-mac-syscall)
            (allow ipc-posix*)
            (allow system-fcntl)
            (allow process-fork)
            (allow process-info* signal)
            (allow mach-lookup)
            (allow darwin-notification-post)
            (allow user-preference-read)
            (allow iokit-get-properties)
            (allow iokit-open-service)

            (allow file-read* file-write* file-clone file-link
                (subpath (param "WORKSPACE"))
            )

            (allow file-read* process-exec
                (subpath (param "WORKSPACE"))
            {runtime_rules}
            )

            (allow file-read* file-map-executable
                (subpath (param "WORKSPACE"))
            {runtime_rules}
            )

            (allow file-read*
                (subpath (param "WORKSPACE"))
            {runtime_rules}
            )
            """
        ).strip()

    def _collect_runtime_read_roots(self, cwd: Path) -> tuple[Path, ...]:
        candidates: set[Path] = {cwd.resolve(), *DEFAULT_MACOS_READ_ROOTS}

        for env_path in os.environ.get("PATH", "").split(os.pathsep):
            if env_path:
                candidates.add(Path(env_path).expanduser())

        interpreter_roots = {
            Path(sys.executable).resolve().parent,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(sys.exec_prefix).resolve(),
            Path(sys.base_exec_prefix).resolve(),
        }
        candidates.update(interpreter_roots)

        for key in ("platlib", "platstdlib", "purelib", "scripts", "stdlib"):
            resolved_path = sysconfig.get_path(key)
            if resolved_path:
                candidates.add(Path(resolved_path).resolve())

        return tuple(sorted(path for path in candidates if path.exists()))

    def _build_environment(self, *, cwd: Path, tmp_dir: Path) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(cwd),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", os.environ.get("LANG", "C.UTF-8")),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TMPDIR": str(tmp_dir),
        }
        if os.environ.get("TERM"):
            env["TERM"] = os.environ["TERM"]
        return env

    def _build_guarded_python_code(self, code: str) -> str:
        return f"{PYTHON_SANDBOX_PREAMBLE}\n\n# User code starts here.\n{code}\n"

    def _should_fallback_from_profile(self, stderr_raw: bytes, exit_code: int) -> bool:
        if exit_code == 0:
            return False
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        fallback_markers = (
            "sandbox_apply: Operation not permitted",
            "sandbox-exec: execvp()",
            "sandbox-exec: execvp(",
        )
        stderr_text_lower = stderr_text.lower()
        return any(marker in stderr_text for marker in fallback_markers) and (
            "operation not permitted" in stderr_text_lower
        )

    def _tokenize_shell_command(self, command: str) -> list[str]:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|;&<>")
        lexer.whitespace_split = True
        return list(lexer)

    def _next_token(self, tokens: list[str], start_index: int) -> str | None:
        for token in tokens[start_index:]:
            if token:
                return token
        return None

    def _looks_like_path(self, token: str) -> bool:
        if not token:
            return False
        if token in {".", "..", "~"}:
            return True
        return token.startswith(("/", "./", "../", "~/")) or "/" in token

    def _is_env_assignment(self, token: str) -> bool:
        if "=" not in token:
            return False
        name, _ = token.split("=", maxsplit=1)
        return bool(name) and name.replace("_", "").isalnum() and not name[0].isdigit()

    def _sbpl_escape(self, path: Path) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')

    def _sbpl_subpath_rule(self, path: Path) -> str:
        return f'    (subpath "{self._sbpl_escape(path)}")'

    def _sbpl_literal_deny_rule(self, path: Path) -> str:
        escaped_path = self._sbpl_escape(path)
        return f'(deny file-read* file-write* (literal "{escaped_path}"))'

    async def _write_temp_script(self, workspace_path: Path, code: str) -> Path:
        def _create_script() -> Path:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="sandbox_exec_",
                dir=workspace_path,
                delete=False,
            ) as handle:
                handle.write(code)
                return Path(handle.name)

        return await asyncio.to_thread(_create_script)
