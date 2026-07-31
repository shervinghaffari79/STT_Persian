import { useState, useRef, useEffect, useMemo } from 'react';
import {
  FileText, Download, Copy, Check, X, Search, Filter,
  Users, Clock, MessageSquare, AlignLeft,
} from 'lucide-react';
import { TranscriptionResult } from '../types';
import { copyText } from '../utils/clipboard';
import SpeakerManager from './SpeakerManager';

// exp(avg_logprob) thresholds for the low/mid/high confidence dot -- these
// are not calibrated probabilities (see asr_engine._confidence), just rough
// bands tuned by eye against fine-tuned Persian Whisper's typical output
// range, where >0.85 is ordinary and <0.6 usually means genuine trouble
// (cross-talk, noise, an unfamiliar word).
function confidenceBand(c: number | null | undefined): 'high' | 'mid' | 'low' | null {
  if (c == null) return null;
  if (c >= 0.85) return 'high';
  if (c >= 0.6) return 'mid';
  return 'low';
}

const CONFIDENCE_DOT: Record<'high' | 'mid' | 'low', string> = {
  high: 'bg-[var(--conf-high)]',
  mid: 'bg-[var(--conf-mid)]',
  low: 'bg-[var(--conf-low)]',
};

// Text colour for the numeric score. The band still drives the colour -- it is
// a useful at-a-glance cue -- but the number itself is now shown, because
// "high/mid/low" collapses the whole 0..1 range into three buckets and a 0.86
// then looks identical to a 1.00.
const CONFIDENCE_TEXT: Record<'high' | 'mid' | 'low', string> = {
  high: 'text-[var(--conf-high)]',
  mid: 'text-[var(--conf-mid)]',
  low: 'text-[var(--conf-low)]',
};

/** exp(avg_logprob) as a percentage. Two significant figures: the underlying
 * number is a mean per-token log-probability pushed through exp(), so it is
 * precise but not accurate, and rendering it to more digits would imply a
 * calibration it does not have. */
const formatConfidence = (c: number) => `${(c * 100).toFixed(0)}%`;

interface TranscriptPanelProps {
  transcription: TranscriptionResult | null;
  currentTime: number;
  onSeek: (time: number) => void;
  onSendToChat: (text: string) => void;
}

// Resolved through CSS variables rather than fixed Tailwind shades, because a
// palette tuned for a near-black page washes out to ~2:1 on a white one. Each
// theme defines its own shade of the same hue in index.css, so a speaker keeps
// a stable identity across themes while staying readable in both.
// Written out in full rather than composed from a template literal: Tailwind's
// JIT scans source text for COMPLETE class strings, so `bg-[var(--spk-${n}-bg)]`
// would never be generated and the colours would silently not apply.
const SPEAKER_COLORS = [
  { bg: 'bg-[var(--spk-1-bg)]', text: 'text-[var(--spk-1-text)]', dot: 'bg-[var(--spk-1-dot)]' },
  { bg: 'bg-[var(--spk-2-bg)]', text: 'text-[var(--spk-2-text)]', dot: 'bg-[var(--spk-2-dot)]' },
  { bg: 'bg-[var(--spk-3-bg)]', text: 'text-[var(--spk-3-text)]', dot: 'bg-[var(--spk-3-dot)]' },
  { bg: 'bg-[var(--spk-4-bg)]', text: 'text-[var(--spk-4-text)]', dot: 'bg-[var(--spk-4-dot)]' },
  { bg: 'bg-[var(--spk-5-bg)]', text: 'text-[var(--spk-5-text)]', dot: 'bg-[var(--spk-5-dot)]' },
] as const;

const SPEAKER_OVERFLOW =
  { bg: 'bg-[var(--spk-x-bg)]', text: 'text-[var(--spk-x-text)]', dot: 'bg-[var(--spk-x-dot)]' } as const;

const getSpeakerColor = (speaker: string) => {
  const n = Number(speaker.replace(/^S/, ''));
  return Number.isFinite(n) && n >= 1 && n <= SPEAKER_COLORS.length
    ? SPEAKER_COLORS[n - 1]
    : SPEAKER_OVERFLOW;  // speakers past the palette share one colour, as before
};

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
  const [copyFailed, setCopyFailed] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [showSpeakerMenu, setShowSpeakerMenu] = useState(false);
  const activeSegmentRef = useRef<HTMLDivElement>(null);

  // Speaker overrides are per-transcription and persisted, because renaming
  // five speakers is real work to redo after a reload. Keyed by transcription
  // id so two files never share each other's names.
  const storageKey = transcription ? `speakers:${transcription.id}` : null;
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
  const [speakerMerges, setSpeakerMerges] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!storageKey) return;
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
      setSpeakerNames(saved.names ?? {});
      setSpeakerMerges(saved.merges ?? {});
    } catch {
      setSpeakerNames({});
      setSpeakerMerges({});
    }
  }, [storageKey]);

  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey,
        JSON.stringify({ names: speakerNames, merges: speakerMerges }));
    } catch {
      // quota or private-mode: renames still work for this session
    }
  }, [storageKey, speakerNames, speakerMerges]);

  // Follow the merge chain to the id a speaker ultimately belongs to. The
  // visited set matters: merging A->B and later B->A is reachable through the
  // UI and would otherwise spin forever.
  const resolveSpeaker = useMemo(() => (id: string): string => {
    const seen = new Set<string>();
    let cur = id;
    while (speakerMerges[cur] && !seen.has(cur)) {
      seen.add(cur);
      cur = speakerMerges[cur];
    }
    return cur;
  }, [speakerMerges]);

  const speakerLabel = (id: string) => {
    const c = resolveSpeaker(id);
    return speakerNames[c] || c;
  };

  const handleMerge = (from: string, into: string) => {
    // resolve the target first, so merging into an already-merged speaker
    // points at the surviving id rather than building a longer chain
    const target = resolveSpeaker(into);
    if (target === from) return;  // would create a cycle
    setSpeakerMerges(m => ({ ...m, [from]: target }));
  };

  const handleUnmerge = (id: string) =>
    setSpeakerMerges(m => { const n = { ...m }; delete n[id]; return n; });

  const handleRename = (id: string, name: string) =>
    setSpeakerNames(n => {
      const next = { ...n };
      if (name) next[id] = name;
      else delete next[id];
      return next;
    });

  // segment counts per canonical speaker, so the manager can show evidence
  const speakerCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const seg of transcription?.segments ?? []) {
      const c = resolveSpeaker(seg.speaker);
      counts[c] = (counts[c] ?? 0) + 1;
    }
    return counts;
  }, [transcription, resolveSpeaker]);

  // speakers still visible after merges, in their original order
  const visibleSpeakers = useMemo(
    () => (transcription?.speakers ?? []).filter(s => resolveSpeaker(s) === s),
    [transcription, resolveSpeaker]);

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

  // Two reasons strict containment (currentTime >= s.start && <= s.end) can
  // match no segment at all, which is exactly "I click and nothing gets
  // selected":
  //  - <audio>.currentTime snaps to the nearest decodable frame for
  //    compressed formats (m4a/mp3/opus), landing tens of ms away from the
  //    exact value a seek requested -- documented browser behaviour, not a
  //    bug in the seek path itself.
  //  - ASR segments are not always back-to-back: a silent gap between two
  //    segments is ordinary, and a slightly-off landing position can fall
  //    exactly in that gap, matching neither neighbour.
  // Scanning from the END first resolves an exact-boundary tie (segment A
  // ends where B starts) in favour of B, the one just entered, rather than
  // the earlier segment that just finished. Falling back to the nearest
  // segment within a short tolerance when nothing contains currentTime
  // covers the frame-snap and gap cases without highlighting something
  // arbitrary when the playhead is genuinely far from any segment.
  const activeSegmentIndex = (() => {
    const segs = transcription?.segments;
    if (!segs || segs.length === 0) return -1;
    for (let i = segs.length - 1; i >= 0; i--) {
      if (currentTime >= segs[i].start && currentTime <= segs[i].end) return i;
    }
    const TOLERANCE = 1.5; // seconds
    let best = -1, bestDist = Infinity;
    segs.forEach((s, i) => {
      const dist = currentTime < s.start ? s.start - currentTime
                 : currentTime - s.end;
      if (dist < bestDist) { bestDist = dist; best = i; }
    });
    return bestDist <= TOLERANCE ? best : -1;
  })();

  useEffect(() => {
    if (activeSegmentRef.current) {
      activeSegmentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeSegmentIndex]);

  const filteredSegments = transcription?.segments.filter(seg => {
    // compare against the resolved id: filtering by a speaker that has absorbed
    // others must show their segments too, not just the ones already labelled
    const matchesSpeaker = filterSpeaker === 'all' || resolveSpeaker(seg.speaker) === filterSpeaker;
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

  const handleCopy = async () => {
    if (!transcription) return;
    const ok = await copyText(transcription.rawText);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } else {
      setCopyFailed(true);
      setTimeout(() => setCopyFailed(false), 2500);
    }
  };

  // Group consecutive same-speaker segments into one paragraph. Without this,
  // "text" view space-joins every VAD-derived ~24s chunk in a row regardless
  // of speaker, which reads as a single undifferentiated wall of text with no
  // paragraph or speaker structure at all.
  // Grouping runs on the RESOLVED speaker, so two ids merged by hand read as
  // one continuous paragraph rather than alternating rows from the same person.
  const paragraphs = (() => {
    // start/end are carried so a paragraph is seekable and can be highlighted
    // as active -- without them this view was the one place in the transcript
    // you could not click to sync the audio.
    const groups: { speaker: string; text: string; start: number; end: number }[] = [];
    for (const seg of filteredSegments) {
      const spk = resolveSpeaker(seg.speaker);
      const last = groups[groups.length - 1];
      if (last && last.speaker === spk) {
        last.text += ' ' + seg.text;
        last.end = seg.end;
      } else {
        groups.push({ speaker: spk, text: seg.text, start: seg.start, end: seg.end });
      }
    }
    return groups;
  })();

  const handleExport = (format: ExportFormat) => {
    if (!transcription) return;
    let content = '';
    let filename = `transcript_${Date.now()}`;
    let mime = 'text/plain';

    // Every export carries the renames and merges. Exporting raw S1/S2 ids
    // would silently discard the work of naming them, which is the main reason
    // anyone renames speakers in the first place.
    switch (format) {
      case 'txt':
        // rebuilt rather than reusing rawText, which the backend wrote with the
        // original ids and knows nothing about later edits
        content = transcription.segments
          .map(s => `[${speakerLabel(s.speaker)}]: ${s.text}`)
          .join('\n\n');
        filename += '.txt';
        break;
      case 'srt':
        content = transcription.segments.map((seg, i) => (
          `${i + 1}\n${formatSRT(seg.start)} --> ${formatSRT(seg.end)}\n[${speakerLabel(seg.speaker)}]: ${seg.text}\n`
        )).join('\n');
        filename += '.srt';
        break;
      case 'json':
        content = JSON.stringify({
          transcription: {
            ...transcription,
            speakers: visibleSpeakers.map(speakerLabel),
            segments: transcription.segments.map(s => ({
              ...s,
              speaker: speakerLabel(s.speaker),
              // keep the machine's own answer alongside the human's, so the
              // export stays a faithful record of what diarization produced
              speakerId: s.speaker,
              words: s.words.map(w => ({ ...w, speaker: speakerLabel(w.speaker) })),
            })),
          },
        }, null, 2);
        filename += '.json';
        mime = 'application/json';
        break;
      case 'csv':
        content = 'Speaker,SpeakerId,Start,End,Text\n' +
          transcription.segments.map(s =>
            `"${speakerLabel(s.speaker).replace(/"/g, '""')}","${s.speaker}","${formatTime(s.start)}","${formatTime(s.end)}","${s.text.replace(/"/g, '""')}"`
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
    // count speakers AFTER merges: the header saying "5 speakers" while the
    // transcript shows three is the first thing a user would notice as wrong
    speakers: visibleSpeakers.length,
    duration: transcription.duration,
  } : null;

  if (!transcription) {
    return (
      <div className="flex flex-col h-full bg-[var(--bg-base)]">
        <div className="panel-header px-4 py-3 flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-violet-500/20 flex items-center justify-center">
            <FileText size={14} className="text-violet-400" />
          </div>
          <span className="text-sm font-semibold text-[var(--text-primary)]">Transcript</span>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 gap-4">
          <div className="w-16 h-16 rounded-2xl bg-[var(--bg-inset)] border border-[var(--border-shell)] flex items-center justify-center">
            <FileText size={28} className="text-[var(--text-muted)]" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-[var(--text-primary)] mb-1">No Transcript Yet</h3>
            <p className="text-sm text-[var(--text-muted)] max-w-[220px]">
              Upload an audio file in the left panel and click Transcribe to get started.
            </p>
          </div>
          <div className="space-y-2 text-left w-full max-w-[220px]">
            {['Speaker diarization', 'Timestamped segments', 'Multi-language support', 'Export to SRT, TXT, JSON'].map(f => (
              <div key={f} className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
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
    <div className="flex flex-col h-full bg-[var(--bg-base)]" onClick={() => { setShowExportMenu(false); setShowFilterMenu(false); setShowSpeakerMenu(false); }}>
      {/* Header */}
      <div className="panel-header px-4 py-3 flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-violet-500/20 flex items-center justify-center">
          <FileText size={14} className="text-violet-400" />
        </div>
        <span className="text-sm font-semibold text-[var(--text-primary)] flex-1">Transcript</span>
        <div className="flex items-center gap-1">
          {/* Copy */}
          <button
            onClick={handleCopy}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-[var(--bg-active)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title={copyFailed ? 'Copy failed -- clipboard unavailable' : 'Copy transcript'}
          >
            {copied
              ? <Check size={14} className="text-green-400" />
              : copyFailed
              ? <X size={14} className="text-red-400" />
              : <Copy size={14} />}
          </button>
          {/* Export */}
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setShowExportMenu(!showExportMenu); }}
              className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-[var(--bg-active)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              title="Export"
            >
              <Download size={14} />
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-1 bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-lg overflow-hidden z-20 shadow-xl w-28" onClick={e => e.stopPropagation()}>
                {(['txt', 'srt', 'json', 'csv'] as ExportFormat[]).map(fmt => (
                  <button
                    key={fmt}
                    onClick={() => handleExport(fmt)}
                    className="w-full text-left px-3 py-2 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-active)] uppercase font-mono"
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
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-[var(--bg-active)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Analyze in chat"
          >
            <MessageSquare size={14} />
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="px-4 py-2.5 bg-[var(--bg-base)] border-b border-[var(--border-shell)] flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
            <Users size={11} />
            <span>{stats.speakers} speakers</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
            <AlignLeft size={11} />
            <span>{stats.words.toLocaleString()} words</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
            <Clock size={11} />
            <span>{formatTime(stats.duration)}</span>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="px-4 py-2 border-b border-[var(--border-shell)] flex items-center gap-2">
        {/* Search */}
        <div className="flex-1 relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search transcript..."
            className="w-full bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-lg pl-7 pr-3 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:border-indigo-500/50 transition-colors"
          />
        </div>

        {/* View mode */}
        <div className="flex bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-lg overflow-hidden">
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
                viewMode === mode ? 'bg-indigo-500 text-white' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
            >
              <Icon size={12} />
            </button>
          ))}
        </div>

        {/* Speaker rename / merge */}
        <div className="relative">
          <button
            onClick={(e) => { e.stopPropagation(); setShowSpeakerMenu(!showSpeakerMenu); setShowFilterMenu(false); }}
            title="Rename or merge speakers"
            aria-label="Rename or merge speakers"
            className={`w-7 h-7 flex items-center justify-center rounded-lg border transition-colors ${
              showSpeakerMenu || Object.keys(speakerNames).length > 0 || Object.keys(speakerMerges).length > 0
                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                : 'border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
            }`}
          >
            <Users size={12} />
          </button>
          {showSpeakerMenu && (
            <div onClick={e => e.stopPropagation()}>
              <SpeakerManager
                allSpeakers={transcription.speakers}
                names={speakerNames}
                counts={speakerCounts}
                getColor={getSpeakerColor}
                resolve={resolveSpeaker}
                onRename={handleRename}
                onMerge={handleMerge}
                onUnmerge={handleUnmerge}
                onClose={() => setShowSpeakerMenu(false)}
              />
            </div>
          )}
        </div>

        {/* Filter */}
        <div className="relative">
          <button
            onClick={(e) => { e.stopPropagation(); setShowFilterMenu(!showFilterMenu); setShowSpeakerMenu(false); }}
            className={`w-7 h-7 flex items-center justify-center rounded-lg border transition-colors ${
              filterSpeaker !== 'all'
                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                : 'border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
            }`}
          >
            <Filter size={12} />
          </button>
          {showFilterMenu && (
            <div className="absolute right-0 top-full mt-1 bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-lg overflow-hidden z-20 shadow-xl min-w-[120px]" onClick={e => e.stopPropagation()}>
              <button
                onClick={() => { setFilterSpeaker('all'); setShowFilterMenu(false); }}
                className={`w-full text-left px-3 py-2 text-xs ${filterSpeaker === 'all' ? 'text-indigo-300 bg-indigo-500/10' : 'text-[var(--text-secondary)] hover:bg-[var(--bg-active)]'}`}
              >
                All Speakers
              </button>
              {visibleSpeakers.map(sp => (
                <button
                  key={sp}
                  onClick={() => { setFilterSpeaker(sp); setShowFilterMenu(false); }}
                  className={`w-full text-left px-3 py-2 text-xs flex items-center gap-2 ${filterSpeaker === sp ? 'text-indigo-300 bg-indigo-500/10' : 'text-[var(--text-secondary)] hover:bg-[var(--bg-active)]'}`}
                >
                  <span className={`w-2 h-2 rounded-full ${getSpeakerColor(sp).dot}`} />
                  {speakerLabel(sp)}
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
              const spk = resolveSpeaker(seg.speaker);
              const colors = getSpeakerColor(spk);
              const isActive = activeSegmentIndex >= 0 && transcription.segments[activeSegmentIndex] === seg;
              const band = confidenceBand(seg.confidence);
              return (
                <div
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  className={`transcript-segment rounded-xl p-3 cursor-pointer transition-all ${
                    isActive ? 'bg-[var(--accent-tint)] border border-[var(--accent-border)]' : 'hover:bg-[var(--bg-active)] border border-transparent'
                  }`}
                  onClick={() => onSeek(seg.start)}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full ${colors.bg}`}>
                      <div className={`w-1.5 h-1.5 rounded-full ${colors.dot}`} />
                      <span className={`text-[11px] font-semibold ${colors.text}`} dir="auto">
                        {speakerNames[spk] ? speakerNames[spk] : `Speaker ${spk}`}
                      </span>
                    </div>
                    <span className="text-[11px] text-[var(--text-muted)]">{formatTime(seg.start)} – {formatTime(seg.end)}</span>
                    {band && seg.confidence != null && (
                      <span
                        className="inline-flex items-center gap-1"
                        title={`ASR confidence ${formatConfidence(seg.confidence)} — exp(avg_logprob) over this segment's tokens. `
                          + `Not a calibrated probability: read it as a relative signal for which segments the model was least sure about.`
                          + (band === 'low' ? ' Worth checking against the audio.' : '')}
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${CONFIDENCE_DOT[band]}`} />
                        <span className={`text-[11px] font-mono tabular-nums ${CONFIDENCE_TEXT[band]}`}>
                          {formatConfidence(seg.confidence)}
                        </span>
                      </span>
                    )}
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
                  <p className="text-sm text-[var(--text-primary)] leading-relaxed text-right" dir="rtl">{highlightText(seg.text)}</p>
                </div>
              );
            })}
          </div>
        )}

        {viewMode === 'text' && (
          <div className="prose max-w-none space-y-3">
            {paragraphs.map((p, idx) => {
              const colors = getSpeakerColor(p.speaker);
              const isActive = currentTime >= p.start && currentTime <= p.end;
              return (
                <p
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  onClick={() => onSeek(p.start)}
                  title={`${formatTime(p.start)} – ${formatTime(p.end)}`}
                  className={`text-sm text-[var(--text-primary)] leading-loose whitespace-pre-wrap
                              text-right cursor-pointer rounded-lg px-2 py-1 transition-colors ${
                                isActive ? 'bg-[var(--accent-tint)]' : 'hover:bg-[var(--bg-active)]'
                              }`}
                  dir="rtl"
                >
                  <span className={`text-[11px] font-semibold ${colors.text} ml-1.5`} dir="auto">
                    {speakerLabel(p.speaker)}:
                  </span>
                  {p.text}
                </p>
              );
            })}
          </div>
        )}

        {viewMode === 'timestamps' && (
          <div className="space-y-1">
            {filteredSegments.map((seg, idx) => {
              const spk = resolveSpeaker(seg.speaker);
              const colors = getSpeakerColor(spk);
              const isActive = activeSegmentIndex >= 0 && transcription.segments[activeSegmentIndex] === seg;
              return (
                <div
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  className={`flex gap-3 text-xs py-1.5 px-2 rounded-lg cursor-pointer transition-colors ${
                    isActive ? 'bg-[var(--accent-tint)]' : 'hover:bg-[var(--bg-active)]'
                  }`}
                  onClick={() => onSeek(seg.start)}
                >
                  <span className="text-indigo-400 font-mono w-12 flex-shrink-0">{formatTime(seg.start)}</span>
                  {/* named speakers need more room than "S1"; truncate rather
                      than let a long name push the text column around */}
                  <span className={`font-semibold w-20 flex-shrink-0 truncate ${colors.text}`}
                        dir="auto" title={speakerLabel(spk)}>{speakerLabel(spk)}</span>
                  <span className="text-[var(--text-secondary)] leading-relaxed text-right flex-1" dir="rtl">{highlightText(seg.text)}</span>
                </div>
              );
            })}
          </div>
        )}

        {filteredSegments.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Search size={24} className="text-[var(--text-muted)] mb-2" />
            <p className="text-sm text-[var(--text-muted)]">No results found</p>
          </div>
        )}
      </div>

      {/* Quick Analysis Buttons */}
      <div className="p-4 border-t border-[var(--border-shell)]">
        <p className="text-xs text-[var(--text-muted)] mb-2">Quick Analysis</p>
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
              className="text-xs px-2.5 py-1 bg-[var(--bg-raised)] hover:bg-[var(--border-shell)] border border-[var(--border-subtle)] hover:border-indigo-500/40 text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg transition-all"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
