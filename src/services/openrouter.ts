import { ChatMessage } from '../types';

const OPENROUTER_KEY = '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';
const BASE_URL = 'https://openrouter.ai/api/v1';
const MODEL = 'arcee-ai/trinity-large-preview:free';

export async function streamChatCompletion(
  messages: ChatMessage[],
  transcriptContext: string,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: string) => void
): Promise<void> {
  const systemPrompt = transcriptContext
    ? `You are a speech analytics AI assistant. You have access to the following transcript from a recorded conversation.

TRANSCRIPT:
${transcriptContext}

Guidelines:
- Always respond in Persian (Farsi) regardless of the language used in the question
- Keep answers brief and to the point
- Reference specific speakers (S1, S2, S3, etc.) when relevant
- Include timestamps only when directly useful`
    : `You are a helpful AI assistant specialized in speech transcription and audio analysis. Always respond in Persian (Farsi) briefly and to the point.`;

  const apiMessages = [
    { role: 'system', content: systemPrompt },
    ...messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role,
        content: m.content,
      })),
  ];

  try {
    const resp = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENROUTER_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': window.location.origin,
        'X-Title': 'VoiceScript AI',
      },
      body: JSON.stringify({
        model: MODEL,
        messages: apiMessages,
        stream: true,
        max_tokens: 2048,
        temperature: 0.7,
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`OpenRouter error ${resp.status}: ${errText}`);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') {
          if (trimmed === 'data: [DONE]') {
            onDone();
            return;
          }
          continue;
        }

        if (trimmed.startsWith('data: ')) {
          try {
            const json = JSON.parse(trimmed.slice(6));
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) onToken(delta);
          } catch {
            // ignore parse errors in stream
          }
        }
      }
    }

    onDone();
  } catch (err: any) {
    onError(err.message || 'Unknown error');
  }
}

export async function generateTitle(transcript: string): Promise<string> {
  try {
    const resp = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENROUTER_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': window.location.origin,
        'X-Title': 'VoiceScript AI',
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: 'user',
            content: `Generate a concise 4-6 word title for this transcript (no quotes, no punctuation at end):\n\n${transcript.slice(0, 500)}`,
          },
        ],
        max_tokens: 20,
        temperature: 0.5,
      }),
    });

    if (!resp.ok) return 'New Analysis';
    const data = await resp.json();
    return data.choices?.[0]?.message?.content?.trim() || 'New Analysis';
  } catch {
    return 'New Analysis';
  }
}
