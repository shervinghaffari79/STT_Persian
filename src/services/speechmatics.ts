import { TranscriptSegment, TranscriptWord } from '../types';

const SPEECHMATICS_KEYS = [
  'mmHW22XmD0ZoeVnoIeG0x5YeSW4kPzYQ',
  'mXgpPUAWVLwLvv847v2X4gDrWoQ3421b',
  'YBczrKzK5EykjzN4uenuxrry0Fu85N9P',
];
let _smKeyIndex = 0;
function pickSmKey(): string {
  const key = SPEECHMATICS_KEYS[_smKeyIndex % SPEECHMATICS_KEYS.length];
  _smKeyIndex++;
  return key;
}
const SM_BASE = 'https://asr.api.speechmatics.com/v2';

export interface SMJob {
  id: string;
  status: 'running' | 'done' | 'rejected' | 'deleted' | 'waiting';
}

export async function submitTranscriptionJob(
  audioFile: File,
  language: string = 'fa',
  onProgress?: (msg: string) => void
): Promise<{ jobId: string; apiKey: string }> {
  onProgress?.('Submitting audio for transcription...');

  const apiKey = pickSmKey();
  const formData = new FormData();

  const config = {
    type: 'transcription',
    transcription_config: {
      language,
      operating_point: 'enhanced',
      diarization: 'speaker',
      output_locale: language === 'en' ? 'en-US' : undefined,
    },
  };

  formData.append('config', JSON.stringify(config));
  formData.append('data_file', audioFile);

  const resp = await fetch(`${SM_BASE}/jobs`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Speechmatics submit failed: ${resp.status} – ${err}`);
  }

  const data = await resp.json();
  return { jobId: data.id as string, apiKey };
}

export async function pollJobStatus(
  jobId: string,
  apiKey: string,
  onProgress?: (msg: string) => void
): Promise<void> {
  let attempts = 0;
  const maxAttempts = 120; // 10 minutes

  while (attempts < maxAttempts) {
    await sleep(5000);
    attempts++;

    const resp = await fetch(`${SM_BASE}/jobs/${jobId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    if (!resp.ok) throw new Error(`Poll failed: ${resp.status}`);

    const data = await resp.json();
    const job: SMJob = data.job;

    onProgress?.(`Processing... (${attempts * 5}s elapsed) Status: ${job.status}`);

    if (job.status === 'done') return;
    if (job.status === 'rejected') throw new Error('Transcription job rejected by Speechmatics');
  }

  throw new Error('Transcription timed out after 10 minutes');
}

export async function fetchTranscriptJSON(jobId: string, apiKey: string): Promise<{
  segments: TranscriptSegment[];
  rawText: string;
  speakers: string[];
}> {
  const resp = await fetch(`${SM_BASE}/jobs/${jobId}/transcript?format=json-v2`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!resp.ok) throw new Error(`Fetch transcript failed: ${resp.status}`);

  const data = await resp.json();
  return parseSmTranscript(data);
}

function parseSmTranscript(data: any): {
  segments: TranscriptSegment[];
  rawText: string;
  speakers: string[];
} {
  const results: any[] = data.results || [];
  const speakerSet = new Set<string>();
  const words: TranscriptWord[] = [];

  for (const item of results) {
    if (item.alternatives && item.alternatives.length > 0) {
      const alt = item.alternatives[0];
      const speaker = alt.speaker || alt.metadata?.speaker || 'S1';
      speakerSet.add(speaker);
      words.push({
        start: item.start_time,
        end: item.end_time,
        text: alt.content,
        speaker,
      });
    }
  }

  // Group words into speaker segments
  const segments: TranscriptSegment[] = [];
  let currentSegment: TranscriptSegment | null = null;

  for (const word of words) {
    // Skip punctuation for segment grouping but still add to words
    const isPunct = /^[.،؟!,?:;]$/.test(word.text);

    if (!currentSegment || (currentSegment.speaker !== word.speaker && !isPunct)) {
      if (currentSegment) segments.push(currentSegment);
      currentSegment = {
        speaker: word.speaker,
        start: word.start,
        end: word.end,
        text: word.text,
        words: [word],
      };
    } else {
      currentSegment.end = word.end;
      currentSegment.text += isPunct ? word.text : ' ' + word.text;
      currentSegment.words.push(word);
    }
  }

  if (currentSegment) segments.push(currentSegment);

  const rawText = segments.map(s => `[${s.speaker}]: ${s.text}`).join('\n\n');
  const speakers = Array.from(speakerSet);

  return { segments, rawText, speakers };
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function generateDemoTranscription(): {
  segments: TranscriptSegment[];
  rawText: string;
  speakers: string[];
} {
  const segments: TranscriptSegment[] = [
    {
      speaker: 'S1',
      start: 0.0,
      end: 5.2,
      text: 'Hello, welcome to our quarterly review meeting. Shall we get started?',
      words: [
        { start: 0.0, end: 0.5, text: 'Hello,', speaker: 'S1' },
        { start: 0.5, end: 0.9, text: 'welcome', speaker: 'S1' },
        { start: 0.9, end: 1.1, text: 'to', speaker: 'S1' },
        { start: 1.1, end: 1.4, text: 'our', speaker: 'S1' },
        { start: 1.4, end: 2.0, text: 'quarterly', speaker: 'S1' },
        { start: 2.0, end: 2.5, text: 'review', speaker: 'S1' },
        { start: 2.5, end: 3.2, text: 'meeting.', speaker: 'S1' },
        { start: 3.2, end: 3.8, text: 'Shall', speaker: 'S1' },
        { start: 3.8, end: 4.0, text: 'we', speaker: 'S1' },
        { start: 4.0, end: 4.3, text: 'get', speaker: 'S1' },
        { start: 4.3, end: 5.2, text: 'started?', speaker: 'S1' },
      ],
    },
    {
      speaker: 'S2',
      start: 5.5,
      end: 11.0,
      text: "Yes, absolutely. I've prepared the slides and the financial report for this quarter.",
      words: [
        { start: 5.5, end: 5.8, text: 'Yes,', speaker: 'S2' },
        { start: 5.8, end: 6.5, text: 'absolutely.', speaker: 'S2' },
        { start: 6.7, end: 6.9, text: "I've", speaker: 'S2' },
        { start: 6.9, end: 7.4, text: 'prepared', speaker: 'S2' },
        { start: 7.4, end: 7.6, text: 'the', speaker: 'S2' },
        { start: 7.6, end: 8.0, text: 'slides', speaker: 'S2' },
        { start: 8.0, end: 8.2, text: 'and', speaker: 'S2' },
        { start: 8.2, end: 8.4, text: 'the', speaker: 'S2' },
        { start: 8.4, end: 9.0, text: 'financial', speaker: 'S2' },
        { start: 9.0, end: 9.5, text: 'report', speaker: 'S2' },
        { start: 9.5, end: 9.7, text: 'for', speaker: 'S2' },
        { start: 9.7, end: 10.0, text: 'this', speaker: 'S2' },
        { start: 10.0, end: 11.0, text: 'quarter.', speaker: 'S2' },
      ],
    },
    {
      speaker: 'S1',
      start: 11.2,
      end: 16.5,
      text: "Great. Let's start with the revenue numbers. How did we perform compared to last quarter?",
      words: [
        { start: 11.2, end: 11.8, text: 'Great.', speaker: 'S1' },
        { start: 12.0, end: 12.3, text: "Let's", speaker: 'S1' },
        { start: 12.3, end: 12.6, text: 'start', speaker: 'S1' },
        { start: 12.6, end: 12.8, text: 'with', speaker: 'S1' },
        { start: 12.8, end: 13.0, text: 'the', speaker: 'S1' },
        { start: 13.0, end: 13.5, text: 'revenue', speaker: 'S1' },
        { start: 13.5, end: 14.0, text: 'numbers.', speaker: 'S1' },
        { start: 14.2, end: 14.5, text: 'How', speaker: 'S1' },
        { start: 14.5, end: 14.7, text: 'did', speaker: 'S1' },
        { start: 14.7, end: 14.9, text: 'we', speaker: 'S1' },
        { start: 14.9, end: 15.3, text: 'perform', speaker: 'S1' },
        { start: 15.3, end: 15.6, text: 'compared', speaker: 'S1' },
        { start: 15.6, end: 15.8, text: 'to', speaker: 'S1' },
        { start: 15.8, end: 16.0, text: 'last', speaker: 'S1' },
        { start: 16.0, end: 16.5, text: 'quarter?', speaker: 'S1' },
      ],
    },
    {
      speaker: 'S2',
      start: 16.8,
      end: 25.0,
      text: "We saw a 23% increase in revenue, reaching $4.2 million. Customer acquisition was up by 18% and churn rate dropped to 2.1%.",
      words: [
        { start: 16.8, end: 17.0, text: 'We', speaker: 'S2' },
        { start: 17.0, end: 17.3, text: 'saw', speaker: 'S2' },
        { start: 17.3, end: 17.5, text: 'a', speaker: 'S2' },
        { start: 17.5, end: 18.0, text: '23%', speaker: 'S2' },
        { start: 18.0, end: 18.6, text: 'increase', speaker: 'S2' },
        { start: 18.6, end: 18.8, text: 'in', speaker: 'S2' },
        { start: 18.8, end: 19.3, text: 'revenue,', speaker: 'S2' },
        { start: 19.3, end: 19.7, text: 'reaching', speaker: 'S2' },
        { start: 19.7, end: 20.5, text: '$4.2', speaker: 'S2' },
        { start: 20.5, end: 21.2, text: 'million.', speaker: 'S2' },
        { start: 21.5, end: 22.0, text: 'Customer', speaker: 'S2' },
        { start: 22.0, end: 22.6, text: 'acquisition', speaker: 'S2' },
        { start: 22.6, end: 22.8, text: 'was', speaker: 'S2' },
        { start: 22.8, end: 23.0, text: 'up', speaker: 'S2' },
        { start: 23.0, end: 23.2, text: 'by', speaker: 'S2' },
        { start: 23.2, end: 23.8, text: '18%', speaker: 'S2' },
        { start: 23.8, end: 24.0, text: 'and', speaker: 'S2' },
        { start: 24.0, end: 24.3, text: 'churn', speaker: 'S2' },
        { start: 24.3, end: 24.6, text: 'rate', speaker: 'S2' },
        { start: 24.6, end: 24.9, text: 'dropped', speaker: 'S2' },
        { start: 24.9, end: 25.0, text: 'to 2.1%.', speaker: 'S2' },
      ],
    },
    {
      speaker: 'S3',
      start: 25.3,
      end: 31.0,
      text: "Those are impressive numbers. What drove the customer acquisition growth? Was it the new marketing campaign?",
      words: [
        { start: 25.3, end: 25.7, text: 'Those', speaker: 'S3' },
        { start: 25.7, end: 25.9, text: 'are', speaker: 'S3' },
        { start: 25.9, end: 26.5, text: 'impressive', speaker: 'S3' },
        { start: 26.5, end: 27.0, text: 'numbers.', speaker: 'S3' },
        { start: 27.2, end: 27.5, text: 'What', speaker: 'S3' },
        { start: 27.5, end: 27.8, text: 'drove', speaker: 'S3' },
        { start: 27.8, end: 28.0, text: 'the', speaker: 'S3' },
        { start: 28.0, end: 28.5, text: 'customer', speaker: 'S3' },
        { start: 28.5, end: 29.0, text: 'acquisition', speaker: 'S3' },
        { start: 29.0, end: 29.5, text: 'growth?', speaker: 'S3' },
        { start: 29.7, end: 29.9, text: 'Was', speaker: 'S3' },
        { start: 29.9, end: 30.1, text: 'it', speaker: 'S3' },
        { start: 30.1, end: 30.3, text: 'the', speaker: 'S3' },
        { start: 30.3, end: 30.6, text: 'new', speaker: 'S3' },
        { start: 30.6, end: 31.0, text: 'marketing campaign?', speaker: 'S3' },
      ],
    },
    {
      speaker: 'S2',
      start: 31.2,
      end: 40.0,
      text: "Primarily yes. The LinkedIn campaign outperformed expectations by 35%. We also benefited from the product feature releases in October which increased organic signups significantly.",
      words: [
        { start: 31.2, end: 32.0, text: 'Primarily', speaker: 'S2' },
        { start: 32.0, end: 32.3, text: 'yes.', speaker: 'S2' },
        { start: 32.5, end: 32.8, text: 'The', speaker: 'S2' },
        { start: 32.8, end: 33.4, text: 'LinkedIn', speaker: 'S2' },
        { start: 33.4, end: 33.9, text: 'campaign', speaker: 'S2' },
        { start: 33.9, end: 34.6, text: 'outperformed', speaker: 'S2' },
        { start: 34.6, end: 35.3, text: 'expectations', speaker: 'S2' },
        { start: 35.3, end: 35.5, text: 'by', speaker: 'S2' },
        { start: 35.5, end: 36.0, text: '35%.', speaker: 'S2' },
        { start: 36.5, end: 36.8, text: 'We', speaker: 'S2' },
        { start: 36.8, end: 37.0, text: 'also', speaker: 'S2' },
        { start: 37.0, end: 37.5, text: 'benefited', speaker: 'S2' },
        { start: 37.5, end: 37.7, text: 'from', speaker: 'S2' },
        { start: 37.7, end: 37.9, text: 'the', speaker: 'S2' },
        { start: 37.9, end: 38.3, text: 'product', speaker: 'S2' },
        { start: 38.3, end: 38.7, text: 'feature', speaker: 'S2' },
        { start: 38.7, end: 39.1, text: 'releases', speaker: 'S2' },
        { start: 39.1, end: 39.3, text: 'in', speaker: 'S2' },
        { start: 39.3, end: 39.7, text: 'October', speaker: 'S2' },
        { start: 39.7, end: 40.0, text: 'which increased organic signups significantly.', speaker: 'S2' },
      ],
    },
  ];

  const rawText = segments.map(s => `[${s.speaker}]: ${s.text}`).join('\n\n');
  return { segments, rawText, speakers: ['S1', 'S2', 'S3'] };
}
