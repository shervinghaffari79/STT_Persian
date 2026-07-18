import { ChatMessage } from '../types';

/**
 * Local chat service — talks to the on-device Qwen3-4B (MLX) via the backend,
 * replacing the cloud OpenRouter service. Same signatures as the old
 * `openrouter.ts` so the ChatPanel needs no other changes.
 *
 * Requests go through Vite's dev proxy: /api -> http://127.0.0.1:8000
 */

const API = '/api';

export async function streamChatCompletion(
  messages: ChatMessage[],
  transcriptContext: string,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const resp = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        messages: messages
          .filter(m => m.role !== 'system')
          .map(m => ({ role: m.role, content: m.content })),
        transcript: transcriptContext || '',
      }),
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`Local chat backend error ${resp.status}. Is backend/server.py running?`);
    }

    // The backend streams raw generated text (token deltas).
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      if (chunk) onToken(chunk);
    }
    onDone();
  } catch (err: any) {
    // a user-initiated abort is a normal stop, not an error
    if (err?.name === 'AbortError' || signal?.aborted) {
      onDone();
      return;
    }
    onError(err.message || 'Local chat failed');
  }
}

export async function generateTitle(transcript: string): Promise<string> {
  try {
    const resp = await fetch(`${API}/chat/title`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript }),
    });
    if (!resp.ok) return 'تحلیل جدید';
    const data = await resp.json();
    return (data.title as string)?.trim() || 'تحلیل جدید';
  } catch {
    return 'تحلیل جدید';
  }
}
