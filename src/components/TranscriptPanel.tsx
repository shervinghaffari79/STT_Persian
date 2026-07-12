import { useState, useRef, useEffect } from 'react';
import {
  FileText, Download, Copy, Check, Search, Filter,
  Users, Clock, MessageSquare, AlignLeft,
} from 'lucide-react';
import { TranscriptionResult } from '../types';

interface TranscriptPanelProps {
  transcription: TranscriptionResult | null;
  currentTime: number;
  onSeek: (time: number) => void;
  onSendToChat: (text: string) => void;
}

const SPEAKER_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  S1: { bg: 'bg-indigo-500/10', text: 'text-indigo-300', dot: 'bg-indigo-500' },
  S2: { bg: 'bg-emerald-500/10', text: 'text-emerald-300', dot: 'bg-emerald-500' },
  S3: { bg: 'bg-amber-500/10', text: 'text-amber-300', dot: 'bg-amber-500' },
  S4: { bg: 'bg-rose-500/10', text: 'text-rose-300', dot: 'bg-rose-500' },
  S5: { bg: 'bg-pink-500/10', text: 'text-pink-300', dot: 'bg-pink-500' },
};

const getSpeakerColor = (speaker: string) =>
  SPEAKER_COLORS[speaker] || { bg: 'bg-purple-500/10', text: 'text-purple-300', dot: 'bg-purple-500' };

type ViewMode = 'speaker' | 'text' | 'timestamps';
type ExportFormat = 'txt' | 'srt' | 'json' | 'csv';

export default function TranscriptPanel({
  transcription,
  currentTime,
  onSeek,
  onSendToChat,
}: TranscriptPanelProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('speaker');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterSpeaker, setFilterSpeaker] = useState<string>('all');
  const [copied, setCopied] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const activeSegmentRef = useRef<HTMLDivElement>(null);

  const formatTime = (t: number) => {
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(t % 60).toString().padStart(2, '0');
    return h > 0 ? `${h}:${m}:${s}` : `${m}:${s}`;
  };

  const formatSRT = (t: number) => {
    const h = Math.floor(t / 3600).toString().padStart(2, '0');
    const m = Math.floor((t % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(t % 60).toString().padStart(2, '0');
    const ms = Math.floor((t % 1) * 1000).toString().padStart(3, '0');
    return `${h}:${m}:${s},${ms}`;
  };

  const activeSegmentIndex = transcription?.segments.findIndex(
    s => currentTime >= s.start && currentTime <= s.end
  ) ?? -1;

  useEffect(() => {
    if (activeSegmentRef.current) {
      activeSegmentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeSegmentIndex]);

  const filteredSegments = transcription?.segments.filter(seg => {
    const matchesSpeaker = filterSpeaker === 'all' || seg.speaker === filterSpeaker;
    const matchesSearch = !searchQuery || seg.text.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSpeaker && matchesSearch;
  }) ?? [];

  const highlightText = (text: string) => {
    if (!searchQuery) return text;
    const parts = text.split(new RegExp(`(${searchQuery})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === searchQuery.toLowerCase()
        ? <mark key={i} className="bg-yellow-400/30 text-yellow-200 rounded px-0.5">{part}</mark>
        : part
    );
  };

  const handleCopy = () => {
    if (!transcription) return;
    navigator.clipboard.writeText(transcription.rawText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExport = (format: ExportFormat) => {
    if (!transcription) return;
    let content = '';
    let filename = `transcript_${Date.now()}`;
    let mime = 'text/plain';

    switch (format) {
      case 'txt':
        content = transcription.rawText;
        filename += '.txt';
        break;
      case 'srt':
        content = transcription.segments.map((seg, i) => (
          `${i + 1}\n${formatSRT(seg.start)} --> ${formatSRT(seg.end)}\n[${seg.speaker}]: ${seg.text}\n`
        )).join('\n');
        filename += '.srt';
        break;
      case 'json':
        content = JSON.stringify({ transcription }, null, 2);
        filename += '.json';
        mime = 'application/json';
        break;
      case 'csv':
        content = 'Speaker,Start,End,Text\n' +
          transcription.segments.map(s =>
            `"${s.speaker}","${formatTime(s.start)}","${formatTime(s.end)}","${s.text.replace(/"/g, '""')}"`
          ).join('\n');
        filename += '.csv';
        mime = 'text/csv';
        break;
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  };

  // Stats
  const stats = transcription ? {
    words: transcription.segments.reduce((acc, s) => acc + s.words.length, 0),
    segments: transcription.segments.length,
    speakers: transcription.speakers.length,
    duration: transcription.duration,
  } : null;

  if (!transcription) {
    return (
      <div className="flex flex-col h-full bg-[#0f0f0f]">
        <div className="panel-header px-4 py-3 flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-violet-500/20 flex items-center justify-center">
            <FileText size={14} className="text-violet-400" />
          </div>
          <span className="text-sm font-semibold text-white">Transcript</span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4">
          <div className="w-16 h-16 rounded-2xl bg-[#161616] border border-[#222] flex items-center justify-center">
            <FileText size={28} className="text-gray-600" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white mb-1">No Transcript Yet</h3>
            <p className="text-sm text-gray-500 max-w-[220px]">
              Upload an audio file in the left panel and click Transcribe to get started.
            </p>
          </div>
          <div className="space-y-2 text-left w-full max-w-[220px]">
            {['Speaker diarization', 'Timestamped segments', 'Multi-language support', 'Export to SRT, TXT, JSON'].map(f => (
              <div key={f} className="flex items-center gap-2 text-xs text-gray-400">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                {f}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]" onClick={() => { setShowExportMenu(false); setShowFilterMenu(false); }}>
      {/* Header */}
      <div className="panel-header px-4 py-3 flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-violet-500/20 flex items-center justify-center">
          <FileText size={14} className="text-violet-400" />
        </div>
        <span className="text-sm font-semibold text-white flex-1">Transcript</span>
        <div className="flex items-center gap-1">
          {/* Copy */}
          <button
            onClick={handleCopy}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            title="Copy transcript"
          >
            {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
          </button>
          {/* Export */}
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setShowExportMenu(!showExportMenu); }}
              className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
              title="Export"
            >
              <Download size={14} />
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-hidden z-20 shadow-xl w-28" onClick={e => e.stopPropagation()}>
                {(['txt', 'srt', 'json', 'csv'] as ExportFormat[]).map(fmt => (
                  <button
                    key={fmt}
                    onClick={() => handleExport(fmt)}
                    className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-white/5 uppercase font-mono"
                  >
                    .{fmt}
                  </button>
                ))}
              </div>
            )}
          </div>
          {/* Send to chat */}
          <button
            onClick={() => onSendToChat(transcription.rawText)}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            title="Analyze in chat"
          >
            <MessageSquare size={14} />
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="px-4 py-2.5 bg-[#111] border-b border-[#1e1e1e] flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-xs text-gray-400">
            <Users size={11} />
            <span>{stats.speakers} speakers</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-gray-400">
            <AlignLeft size={11} />
            <span>{stats.words.toLocaleString()} words</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-gray-400">
            <Clock size={11} />
            <span>{formatTime(stats.duration)}</span>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="px-4 py-2 border-b border-[#1e1e1e] flex items-center gap-2">
        {/* Search */}
        <div className="flex-1 relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search transcript..."
            className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg pl-7 pr-3 py-1.5 text-xs text-white placeholder-gray-600 focus:border-indigo-500/50 transition-colors"
          />
        </div>

        {/* View mode */}
        <div className="flex bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-hidden">
          {[
            { mode: 'speaker' as ViewMode, icon: Users, title: 'Speaker view' },
            { mode: 'text' as ViewMode, icon: AlignLeft, title: 'Text view' },
            { mode: 'timestamps' as ViewMode, icon: Clock, title: 'Timestamp view' },
          ].map(({ mode, icon: Icon, title }) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              title={title}
              className={`w-7 h-7 flex items-center justify-center transition-colors ${
                viewMode === mode ? 'bg-indigo-500 text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <Icon size={12} />
            </button>
          ))}
        </div>

        {/* Filter */}
        <div className="relative">
          <button
            onClick={(e) => { e.stopPropagation(); setShowFilterMenu(!showFilterMenu); }}
            className={`w-7 h-7 flex items-center justify-center rounded-lg border transition-colors ${
              filterSpeaker !== 'all'
                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                : 'border-[#2a2a2a] text-gray-500 hover:text-gray-300'
            }`}
          >
            <Filter size={12} />
          </button>
          {showFilterMenu && (
            <div className="absolute right-0 top-full mt-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-hidden z-20 shadow-xl min-w-[120px]" onClick={e => e.stopPropagation()}>
              <button
                onClick={() => { setFilterSpeaker('all'); setShowFilterMenu(false); }}
                className={`w-full text-left px-3 py-2 text-xs ${filterSpeaker === 'all' ? 'text-indigo-300 bg-indigo-500/10' : 'text-gray-300 hover:bg-white/5'}`}
              >
                All Speakers
              </button>
              {transcription.speakers.map(sp => (
                <button
                  key={sp}
                  onClick={() => { setFilterSpeaker(sp); setShowFilterMenu(false); }}
                  className={`w-full text-left px-3 py-2 text-xs flex items-center gap-2 ${filterSpeaker === sp ? 'text-indigo-300 bg-indigo-500/10' : 'text-gray-300 hover:bg-white/5'}`}
                >
                  <span className={`w-2 h-2 rounded-full ${getSpeakerColor(sp).dot}`} />
                  Speaker {sp}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Transcript Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-1">
        {viewMode === 'speaker' && (
          <div className="space-y-4">
            {filteredSegments.map((seg, idx) => {
              const colors = getSpeakerColor(seg.speaker);
              const isActive = activeSegmentIndex >= 0 && transcription.segments[activeSegmentIndex] === seg;
              return (
                <div
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  className={`transcript-segment rounded-xl p-3 cursor-pointer transition-all ${
                    isActive ? 'bg-indigo-500/10 border border-indigo-500/30' : 'hover:bg-white/[0.02] border border-transparent'
                  }`}
                  onClick={() => onSeek(seg.start)}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full ${colors.bg}`}>
                      <div className={`w-1.5 h-1.5 rounded-full ${colors.dot}`} />
                      <span className={`text-[11px] font-semibold ${colors.text}`}>Speaker {seg.speaker}</span>
                    </div>
                    <span className="text-[11px] text-gray-600">{formatTime(seg.start)} – {formatTime(seg.end)}</span>
                    {isActive && (
                      <div className="flex gap-0.5 items-end h-3 ml-auto">
                        {[1, 2, 3].map(i => (
                          <div
                            key={i}
                            className="w-0.5 bg-indigo-400 rounded-full animate-pulse"
                            style={{ height: `${40 + i * 20}%`, animationDelay: `${i * 0.15}s` }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-200 leading-relaxed text-right" dir="rtl">{highlightText(seg.text)}</p>
                </div>
              );
            })}
          </div>
        )}

        {viewMode === 'text' && (
          <div className="prose max-w-none">
            <p className="text-sm text-gray-200 leading-loose whitespace-pre-wrap text-right" dir="rtl">
              {filteredSegments.map(seg => seg.text).join(' ')}
            </p>
          </div>
        )}

        {viewMode === 'timestamps' && (
          <div className="space-y-1">
            {filteredSegments.map((seg, idx) => {
              const colors = getSpeakerColor(seg.speaker);
              const isActive = activeSegmentIndex >= 0 && transcription.segments[activeSegmentIndex] === seg;
              return (
                <div
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  className={`flex gap-3 text-xs py-1.5 px-2 rounded-lg cursor-pointer transition-colors ${
                    isActive ? 'bg-indigo-500/10' : 'hover:bg-white/[0.02]'
                  }`}
                  onClick={() => onSeek(seg.start)}
                >
                  <span className="text-indigo-400 font-mono w-12 flex-shrink-0">{formatTime(seg.start)}</span>
                  <span className={`font-semibold w-8 flex-shrink-0 ${colors.text}`}>{seg.speaker}</span>
                  <span className="text-gray-300 leading-relaxed text-right flex-1" dir="rtl">{highlightText(seg.text)}</span>
                </div>
              );
            })}
          </div>
        )}

        {filteredSegments.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Search size={24} className="text-gray-600 mb-2" />
            <p className="text-sm text-gray-500">No results found</p>
          </div>
        )}
      </div>

      {/* Quick Analysis Buttons */}
      <div className="p-4 border-t border-[#1e1e1e]">
        <p className="text-xs text-gray-500 mb-2">Quick Analysis</p>
        <div className="flex flex-wrap gap-1.5">
          {[
            'Summarize this transcript',
            'Extract action items',
            'Sentiment analysis',
            'Key topics discussed',
          ].map(prompt => (
            <button
              key={prompt}
              onClick={() => onSendToChat(prompt)}
              className="text-xs px-2.5 py-1 bg-[#1a1a1a] hover:bg-[#222] border border-[#2a2a2a] hover:border-indigo-500/40 text-gray-400 hover:text-white rounded-lg transition-all"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
