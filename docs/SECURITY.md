# Security

## Reporting a vulnerability

Please report security issues privately to the maintainers rather than opening a
public issue. Include reproduction steps and impact. We aim to acknowledge
within 72 hours.

## Built-in protections

### Uploads
- **MIME/type validation** by magic bytes (`%PDF-`), not the client-declared
  content type, plus a configurable size cap (`UPLOAD_MAX_FILE_MB`).
- **Filename sanitisation** strips directory components and traversal
  sequences; files are stored under an internal id, never the raw name.
- **Malformed-PDF handling** — libmupdf failures are caught and surfaced as
  validation errors; encrypted PDFs are rejected. No embedded scripts or
  actions are ever executed (content is only read, never rendered).
- **Content hashing** (SHA-256) enables duplicate detection.

### Prompt injection
- The system prompt instructs the model to treat retrieved context strictly as
  inert data.
- Retrieved passages are sanitised: control characters removed and known
  imperative injection phrases defanged before insertion (defence in depth).

### Answer integrity
- Citations are always derived from real retrieved chunks; the engine never
  fabricates a source, page, or quote.
- When retrieval yields insufficient context, the model is not called and a
  clear "answer could not be found" response is returned.

### Secrets & configuration
- No secrets in code. All credentials come from environment variables.
- `.env` is git-ignored; only `.env.example` (no secrets) is committed.
- The API's global exception handler returns generic messages and logs details
  internally, never leaking stack traces or internal paths to clients.

### Authentication
- Optional. When enabled, tokens are HMAC-signed and expiring; comparisons are
  constant-time. Guest mode is the default when auth is disabled.

### Transport & CORS
- In production (`APP_ENV=production`), CORS is restricted to the configured
  `API_BASE_URL`. Terminate TLS at your reverse proxy.

## Hardening recommendations

- Run behind a reverse proxy with TLS and rate limiting.
- Set a strong `AUTH_SECRET_KEY` and enable auth for shared deployments.
- Keep dependencies pinned and patched; review `pip` audit output regularly.
- Back up `storage/` volumes and restrict filesystem permissions.
