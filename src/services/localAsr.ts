import { TranscriptSegment } from '../types';

/**
 * Client for the local Persian SOTA ASR backend (backend/server.py).
 * Replaces the cloud Speechmatics service with the on-device pipeline:
 *   MLX 8-bit Whisper large-v3 (fa) + Silero VAD + speaker diarization + Hazm.
 *
 * Requests go through Vite's dev proxy: /api -> http://127.0.0.1:8000
 */

const API = '/api';

export interface LocalTranscript {
  duration: number;
  language: string;
  segments: TranscriptSegment[];
  rawText: string;
  speakers: string[];
  processingTime: number;
  fileName?: string;
}

function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

async function submit(file: File, diarize: boolean): Promise<string> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('diarize', diarize ? 'true' : 'false');
  const resp = await fetch(`${API}/transcribe`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(`Backend rejected upload (${resp.status}). Is backend/server.py running on :8000?`);
  const data = await resp.json();
  return data.job_id as string;
}

/**
 * Upload + poll the local backend to completion.
 * `onProgress(message, percent)` — job status updates.
 * `onPartial(segments, speakers)` — growing transcript snapshot for live display.
 */
export async function transcribeLocal(
  file: File,
  opts: { diarize?: boolean } = {},
  onProgress?: (msg: string, pct: number) => void,
  onPartial?: (segments: TranscriptSegment[], speakers: string[]) => void,
): Promise<LocalTranscript> {
  onProgress?.('Uploading to local model…', 3);
  const jobId = await submit(file, opts.diarize ?? true);

  // Segments accumulate here rather than being re-sent whole on every poll:
  // the backend returns only what is newer than `since` (see /api/status), so
  // a long transcript stops costing a full re-serialize + re-parse of every
  // segment and its per-word timings once a second. Callers still receive the
  // complete array, so nothing downstream of this function changes.
  let acc: TranscriptSegment[] = [];
  const maxAttempts = 2400; // ~2h ceiling for very long audio
  for (let i = 0; i < maxAttempts; i++) {
    await sleep(1200);
    // Re-fetch from the LAST turn, not past it. The backend grows the current
    // turn in place while one speaker holds the floor (see pipeline._stream),
    // so the tail of `acc` is not final until somebody else starts talking.
    // Asking from acc.length would skip that growth and the turn would stay
    // frozen at its first sentence until the speaker changed.
    const since = Math.max(0, acc.length - 1);
    const resp = await fetch(`${API}/status/${jobId}?since=${since}`);
    if (!resp.ok) throw new Error(`Status check failed (${resp.status})`);
    const data = await resp.json();

    onProgress?.(data.message || 'Processing…', typeof data.progress === 'number' ? data.progress : 50);

    // stream turns to the UI as they arrive; `fresh` starts at `since`, so it
    // re-states the possibly-grown last turn and then adds any new ones
    const fresh = (data.partial || []) as TranscriptSegment[];
    if (fresh.length) {
      const next = acc.slice(0, since).concat(fresh);
      // partial_total is the backend's authoritative count. A disagreement
      // means our view drifted (a restarted job reusing the id, an older
      // backend ignoring `since`) -- trust the backend rather than build on a
      // bad prefix.
      const total = data.partial_total;
      acc = typeof total === 'number' && total !== next.length ? fresh : next;
      onPartial?.(acc, (data.speakers || []) as string[]);
    }

    if (data.state === 'done') return data.result as LocalTranscript;
    if (data.state === 'error') throw new Error(data.error || 'Transcription failed on the backend');
  }
  throw new Error('Transcription timed out');
}

export async function backendHealth(): Promise<{ status: string; model: string; model_present: boolean }> {
  const resp = await fetch(`${API}/health`);
  if (!resp.ok) throw new Error('backend unreachable');
  return resp.json();
}
