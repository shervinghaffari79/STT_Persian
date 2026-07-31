import { useState, useCallback, useRef } from 'react';
import AudioPanel from './components/AudioPanel';
import TranscriptPanel from './components/TranscriptPanel';
import ChatPanel from './components/ChatPanel';
import { TranscriptionResult } from './types';
import { Mic2, Sun, Moon } from 'lucide-react';
import { useTheme } from './hooks/useTheme';

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [_audioDuration, setAudioDuration] = useState(0);
  
  const [externalChatMessage, setExternalChatMessage] = useState('');
  const [activeMobilePanel, setActiveMobilePanel] = useState<'audio' | 'transcript' | 'chat'>('audio');

  const handleTranscriptionComplete = useCallback((result: TranscriptionResult) => {
    setTranscription(result);
  }, []);

  // live partial transcript while the backend streams segments
  const handlePartialTranscription = useCallback((partial: TranscriptionResult) => {
    setTranscription(partial);
  }, []);

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleAudioLoaded = useCallback((duration: number) => {
    setAudioDuration(duration);
  }, []);

  // A seek is a COMMAND, and must not be conflated with currentTime, which is
  // a continuously-updated REPORT of where playback is. AudioPanel used to
  // detect seeks by watching currentTime and ignoring changes under 0.5s --
  // needed to break the feedback loop against its own onTimeUpdate, but it
  // also silently swallowed real seeks: clicking a transcript segment that
  // starts within 0.5s of the playhead did nothing, and clicking the SAME
  // segment twice did nothing the second time because the state value never
  // changed. Word-level diarization made this far more visible by producing
  // many short, closely-spaced segments.
  //
  // The nonce makes every request distinct, so repeated or nearby seeks all
  // fire, and AudioPanel no longer has to guess which currentTime changes
  // were externally commanded.
  const [seekRequest, setSeekRequest] = useState<{ time: number; nonce: number } | null>(null);
  const seekNonce = useRef(0);

  const handleSeek = useCallback((time: number) => {
    seekNonce.current += 1;
    setSeekRequest({ time, nonce: seekNonce.current });
    // update immediately so the transcript's active-segment highlight and
    // auto-scroll respond on click, rather than waiting for the audio
    // element's next timeupdate event
    setCurrentTime(time);
  }, []);

  const handleSendToChat = useCallback((text: string) => {
    setExternalChatMessage(text);
    setActiveMobilePanel('chat');
  }, []);

  const handleExternalMessageUsed = useCallback(() => {
    setExternalChatMessage('');
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen bg-[var(--bg-shell)] overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="flex items-center px-4 h-12 border-b border-[var(--border-shell)] bg-[var(--bg-shell)] flex-shrink-0 z-10">
        {/* Logo */}
        <div className="flex items-center gap-2.5 mr-6">
          <div className="w-7 h-7 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-lg flex items-center justify-center">
            {/* stays white: it sits on the indigo/violet gradient in both themes */}
            <Mic2 size={14} className="text-white" />
          </div>
          <span className="text-sm font-bold text-[var(--text-primary)] tracking-tight">VoiceScript</span>
          <span className="text-[10px] px-1.5 py-0.5 bg-[var(--accent-tint-strong)] text-[var(--accent-text)] rounded-md font-semibold">AI</span>
        </div>

        {/* Panel Labels (desktop) */}
        <div className="hidden md:flex items-center gap-1 text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[var(--bg-inset)]">
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
            Audio Input
          </div>
          <div className="w-3 h-px bg-[var(--border-strong)]" />
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[var(--bg-inset)]">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-400" />
            Transcript
          </div>
          <div className="w-3 h-px bg-[var(--border-strong)]" />
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[var(--bg-inset)]">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            AI Analysis
          </div>
        </div>

        <div className="flex-1" />

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="w-7 h-7 mr-2 flex items-center justify-center rounded-lg border border-[var(--border-subtle)]
                     text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]
                     transition-colors flex-shrink-0"
        >
          {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
        </button>

        {/* Status */}
        {transcription && (
          <div className="hidden md:flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Transcript ready · {transcription.speakers.length} speakers
          </div>
        )}

        {/* Mobile panel switcher */}
        <div className="md:hidden flex items-center gap-1 bg-[var(--bg-inset)] rounded-xl p-1">
          {[
            { id: 'audio' as const, label: 'Audio', color: 'bg-indigo-500' },
            { id: 'transcript' as const, label: 'Text', color: 'bg-violet-500' },
            { id: 'chat' as const, label: 'Chat', color: 'bg-emerald-500' },
          ].map(panel => (
            <button
              key={panel.id}
              onClick={() => setActiveMobilePanel(panel.id)}
              className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeMobilePanel === panel.id
                  ? `${panel.color} text-white`
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
            >
              {panel.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main Three-Panel Layout */}
      <main className="flex-1 flex overflow-hidden">
        {/* ── LEFT PANEL: Audio ── */}
        <div className={`
          flex-shrink-0 border-r border-[var(--border-shell)] overflow-hidden
          ${activeMobilePanel === 'audio' ? 'flex' : 'hidden'}
          md:flex md:w-[280px] lg:w-[300px] xl:w-[320px]
          flex-col
        `}>
          <AudioPanel
            onTranscriptionComplete={handleTranscriptionComplete}
            onPartialTranscription={handlePartialTranscription}
            seekRequest={seekRequest}
            onTimeUpdate={handleTimeUpdate}
            onAudioLoaded={handleAudioLoaded}
            transcription={transcription}
          />
        </div>

        {/* ── MIDDLE PANEL: Transcript ── */}
        <div className={`
          flex-1 border-r border-[var(--border-shell)] overflow-hidden
          ${activeMobilePanel === 'transcript' ? 'flex' : 'hidden'}
          md:flex
          flex-col
        `}>
          <TranscriptPanel
            transcription={transcription}
            currentTime={currentTime}
            onSeek={handleSeek}
            onSendToChat={handleSendToChat}
          />
        </div>

        {/* ── RIGHT PANEL: Chat ── */}
        <div className={`
          flex-shrink-0 overflow-hidden
          ${activeMobilePanel === 'chat' ? 'flex' : 'hidden'}
          md:flex md:w-[320px] lg:w-[360px] xl:w-[400px]
          flex-col
        `}>
          <ChatPanel
            transcription={transcription}
            externalMessage={externalChatMessage}
            onExternalMessageUsed={handleExternalMessageUsed}
            onSeek={handleSeek}
          />
        </div>
      </main>

    </div>
  );
}
