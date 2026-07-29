/**
 * Copy-to-clipboard that actually reports failure instead of doing nothing.
 *
 * `navigator.clipboard` only exists in a secure context (HTTPS, or localhost).
 * This app's frontend binds to 0.0.0.0 for LAN access (see vite.config.ts /
 * run.ps1 / run.sh) and is typically opened as plain http://<lan-ip>:5000 --
 * a non-secure context on every browser that implements the spec. In that
 * case `navigator.clipboard` is `undefined`, and the previous call sites did
 * `navigator.clipboard.writeText(...)` with no guard: a TypeError thrown
 * synchronously, uncaught, so the click visibly did nothing.
 *
 * Falls back to the legacy execCommand('copy') path, which has no secure-
 * context requirement, then reports true/false so the caller can show an
 * error instead of a silent no-op either way.
 */
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to the legacy path -- some browsers expose the API but
      // still deny the write (e.g. missing permission in an embedded frame)
    }
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
