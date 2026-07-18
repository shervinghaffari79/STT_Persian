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

  let lastCount = 0;
  const maxAttempts = 2400; // ~2h ceiling for very long audio
  for (let i = 0; i < maxAttempts; i++) {
    await sleep(1200);
    const resp = await fetch(`${API}/status/${jobId}`);
    if (!resp.ok) throw new Error(`Status check failed (${resp.status})`);
    const data = await resp.json();

    onProgress?.(data.message || 'Processing…', typeof data.progress === 'number' ? data.progress : 50);

    // stream newly-produced segments to the UI as they arrive
    const partial = (data.partial || []) as TranscriptSegment[];
    if (onPartial && partial.length !== lastCount) {
      lastCount = partial.length;
      onPartial(partial, (data.speakers || []) as string[]);
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
