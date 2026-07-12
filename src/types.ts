export interface TranscriptWord {
  start: number;
  end: number;
  text: string;
  speaker: string;
}

export interface TranscriptSegment {
  speaker: string;
  start: number;
  end: number;
  text: string;
  words: TranscriptWord[];
}

export interface TranscriptionResult {
  id: string;
  fileName: string;
  duration: number;
  language: string;
  createdAt: Date;
  segments: TranscriptSegment[];
  rawText: string;
  speakers: string[];
  processingTime: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning?: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  transcriptionId?: string;
  createdAt: Date;
  updatedAt: Date;
}

export type PanelView = 'audio' | 'transcript' | 'chat';

export interface AudioState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  playbackRate: number;
}
