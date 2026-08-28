"""Mapping of user-facing language names to solidlsp language-server identifiers.

solidlsp identifiers are the values of ``solidlsp.ls_config.LanguageServerId``
(e.g. "python_pyrefly", "typescript", "java", "go", "rust", "cpp").
This module deliberately avoids importing solidlsp so the thin client stays fast;
identifiers are plain strings validated by the daemon.
"""

from __future__ import annotations

import os
import shutil
import sys

# canonical user-facing language -> default solidlsp LanguageServerId value
_PYTHON_SERVER_TO_LS_ID = {
    "pyrefly": "python_pyrefly",
    "basedpyright": "python_basedpyright",
    "pyright": "python",
    "jedi": "python_jedi",
    "ty": "python_ty",
}

_LANGUAGE_ALIASES = {
    "c": "cpp",
    "c++": "cpp",
    "js": "typescript",
    "javascript": "typescript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "rs": "rust",
}

_SUPPORTED_LANGUAGES = ("python", "typescript", "java", "go", "rust", "cpp")

# All solidlsp ids we allow as a direct --language / LSP_TOOL_LANGUAGE value.
_DIRECT_LS_IDS = set(_PYTHON_SERVER_TO_LS_ID.values()) | set(_SUPPORTED_LANGUAGES)

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}
for _ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"):
    _EXT_TO_LANGUAGE[_ext] = "typescript"
for _ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".inl"):
    _EXT_TO_LANGUAGE[_ext] = "cpp"


class LanguageResolutionError(Exception):
    pass


def _python_ls_id() -> str:
    server = os.environ.get("LSP_TOOL_PYTHON_SERVER", "pyrefly").strip().lower()
    try:
        return _PYTHON_SERVER_TO_LS_ID[server]
    except KeyError:
        raise LanguageResolutionError(
            f"unknown LSP_TOOL_PYTHON_SERVER '{server}' (choose from: {', '.join(sorted(_PYTHON_SERVER_TO_LS_ID))})"
        ) from None


def language_to_ls_id(language: str) -> str:
    """Map a user-facing language name to a solidlsp LanguageServerId value."""
    lang = language.strip().lower()
    lang = _LANGUAGE_ALIASES.get(lang, lang)
    if lang == "python":
        return _python_ls_id()
    if lang in _SUPPORTED_LANGUAGES:
        return lang  # for these, the LanguageServerId value equals the language name
    if lang in _DIRECT_LS_IDS:
        return lang  # allow passing a raw solidlsp id such as "python_basedpyright"
    raise LanguageResolutionError(
        f"unsupported language '{language}' (supported: {', '.join(_SUPPORTED_LANGUAGES)})"
    )


def language_from_file(path: str) -> str | None:
    _, ext = os.path.splitext(path)
    return _EXT_TO_LANGUAGE.get(ext.lower())


_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", "build", "dist", ".idea", ".vscode",
}


def language_from_workspace(workspace: str, max_files: int = 4000) -> str | None:
    """Guess the dominant language of a workspace by counting source-file extensions."""
    counts: dict[str, int] = {}
    seen = 0
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            lang = language_from_file(fn)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
            seen += 1
            if seen >= max_files:
                break
        if seen >= max_files:
            break
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def resolve_ls_id(explicit_language: str | None, file_path: str | None, workspace: str) -> str:
    """Resolve the solidlsp LanguageServerId value for a CLI invocation.

    Order: --language flag, LSP_TOOL_LANGUAGE env, the file's extension,
    then the dominant source language of the workspace.
    """
    lang = explicit_language or os.environ.get("LSP_TOOL_LANGUAGE")
    if not lang and file_path:
        lang = language_from_file(file_path)
    if not lang:
        lang = language_from_workspace(workspace)
    if not lang:
        raise LanguageResolutionError(
            "could not determine language; pass --language or set LSP_TOOL_LANGUAGE"
        )
    return language_to_ls_id(lang)


def find_executable(name: str) -> str | None:
    """Find an executable, preferring the directory of the running interpreter (venv bin)."""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(exe_dir, name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which(name)
