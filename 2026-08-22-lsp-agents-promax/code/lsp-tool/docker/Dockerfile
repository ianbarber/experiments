ARG BASE
FROM ${BASE}
COPY lsp-tool /tmp/lsp-tool
# Some ProMax images bake in an unreachable corporate proxy (sys-proxy-rd-relay.byted.org);
# neutralize proxies for the BUILD STEPS ONLY via `env -u` — no ENV changes, so the final
# image environment stays byte-identical to the base apart from the LSP_TOOL_* vars below.
RUN env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u ftp_proxy \
    sh -c 'command -v uv >/dev/null 2>&1 || \
      (command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | sh) || \
      (command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh)'
RUN env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u ftp_proxy \
    PATH="/root/.local/bin:${PATH}" bash /tmp/lsp-tool/install.sh && rm -rf /tmp/lsp-tool
ENV LSP_TOOL_WORKSPACE=/testbed
ENV LSP_TOOL_LANGUAGE=python
# Best-effort pre-warm: auto-starts the daemon, forces first-run setup and an index pass
# into the image layer; never fails the build. (Local venv servers — no network needed.)
RUN cd /testbed && (timeout 300 lsp sym __warmup__ || true) && (lsp daemon stop || true)
