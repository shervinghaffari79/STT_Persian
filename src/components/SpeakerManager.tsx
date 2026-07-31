import { useState } from 'react';
import { Users, Pencil, Check, X, User, ChevronRight } from 'lucide-react';

/**
 * Speaker rename + merge panel.
 *
 * Diarization hands back anonymous cluster ids (S1, S2, …) whose numbering is
 * arbitrary — it reflects the order speakers happened to be encountered, not
 * anything about them. Two things routinely need fixing by hand afterwards:
 *
 *  - the ids mean nothing to a reader, so a meeting transcript is far more
 *    useful once S1 becomes "Parisa"
 *  - clustering over-segments, especially when someone's voice changes across a
 *    long recording (moving closer to the mic, raising their voice). One person
 *    then occupies two ids, and no amount of backend tuning reliably fixes it
 *    for every file — but a human can see it instantly.
 *
 * Merging is stored as a redirect (from → into) rather than by rewriting
 * segments, so it stays reversible and the underlying diarization output is
 * never destroyed.
 */

export interface SpeakerColors {
  bg: string;
  text: string;
  dot: string;
}

interface SpeakerManagerProps {
  /** every speaker id the transcription produced, pre-merge */
  allSpeakers: string[];
  /** id -> display name override */
  names: Record<string, string>;
  /** segment count per id, shown so a merge decision has evidence behind it */
  counts: Record<string, number>;
  getColor: (id: string) => SpeakerColors;
  resolve: (id: string) => string;
  onRename: (id: string, name: string) => void;
  onMerge: (from: string, into: string) => void;
  onUnmerge: (id: string) => void;
  onClose: () => void;
}

export default function SpeakerManager({
  allSpeakers, names, counts, getColor, resolve,
  onRename, onMerge, onUnmerge, onClose,
}: SpeakerManagerProps) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [mergeSource, setMergeSource] = useState<string | null>(null);

  const canonical = allSpeakers.filter(s => resolve(s) === s);
  const mergedAway = allSpeakers.filter(s => resolve(s) !== s);

  const startEdit = (id: string) => {
    setEditing(id);
    setDraft(names[id] ?? '');
    setMergeSource(null);
  };

  const commitEdit = (id: string) => {
    onRename(id, draft.trim());
    setEditing(null);
    setDraft('');
  };

  return (
    <div className="absolute right-0 top-full mt-1 w-72 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-inset)] shadow-xl z-30 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border-subtle)]">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)]">
          <Users size={13} /> Speakers
        </span>
        <button onClick={onClose} className="p-0.5 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)]"
                aria-label="Close speaker manager">
          <X size={13} />
        </button>
      </div>

      <div className="max-h-72 overflow-y-auto py-1">
        {canonical.map(id => {
          const c = getColor(id);
          const absorbed = allSpeakers.filter(s => s !== id && resolve(s) === id);
          const total = counts[id] ?? 0;

          return (
            <div key={id} className="px-3 py-1.5 hover:bg-[var(--bg-active)]">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full shrink-0 ${c.dot}`} />

                {editing === id ? (
                  <>
                    <input
                      autoFocus
                      value={draft}
                      onChange={e => setDraft(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter') commitEdit(id);
                        if (e.key === 'Escape') { setEditing(null); setDraft(''); }
                      }}
                      placeholder={id}
                      dir="auto"
                      className="flex-1 min-w-0 bg-[var(--bg-base)] border border-[var(--border-strong)] rounded px-1.5 py-0.5
                                 text-xs text-[var(--text-primary)] focus:border-indigo-500"
                    />
                    <button onClick={() => commitEdit(id)} aria-label="Save name"
                            className="p-0.5 rounded hover:bg-[var(--bg-hover)] text-emerald-400">
                      <Check size={13} />
                    </button>
                    <button onClick={() => { setEditing(null); setDraft(''); }} aria-label="Cancel rename"
                            className="p-0.5 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)]">
                      <X size={13} />
                    </button>
                  </>
                ) : (
                  <>
                    <span className={`text-xs font-medium truncate ${c.text}`} dir="auto">
                      {names[id] || id}
                    </span>
                    {names[id] && (
                      <span className="text-[10px] text-[var(--text-muted)] shrink-0">{id}</span>
                    )}
                    <span className="ml-auto text-[10px] text-[var(--text-muted)] shrink-0">
                      {total} seg{total === 1 ? '' : 's'}
                    </span>
                    <button onClick={() => startEdit(id)} aria-label={`Rename ${id}`}
                            className="p-0.5 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] shrink-0">
                      <Pencil size={12} />
                    </button>
                  </>
                )}
              </div>

              {absorbed.length > 0 && (
                <div className="mt-1 ml-4 flex flex-wrap gap-1">
                  {absorbed.map(a => (
                    <button
                      key={a}
                      onClick={() => onUnmerge(a)}
                      title={`Split ${a} back out of ${names[id] || id}`}
                      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded
                                 bg-[var(--bg-active)] text-[var(--text-muted)] hover:text-[var(--text-primary)]
                                 hover:bg-[var(--bg-hover)]"
                    >
                      {a} <X size={9} />
                    </button>
                  ))}
                </div>
              )}

              {/* merge picker: choosing a target folds THIS speaker into it */}
              {mergeSource === id ? (
                <div className="mt-1.5 ml-4">
                  <div className="text-[10px] text-[var(--text-muted)] mb-1">Merge into…</div>
                  <div className="flex flex-wrap gap-1">
                    {canonical.filter(o => o !== id).map(o => (
                      <button
                        key={o}
                        onClick={() => { onMerge(id, o); setMergeSource(null); }}
                        className={`text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-active)] hover:bg-[var(--bg-hover)] ${getColor(o).text}`}
                      >
                        {names[o] || o}
                      </button>
                    ))}
                    <button onClick={() => setMergeSource(null)}
                            className="text-[10px] px-1.5 py-0.5 rounded text-[var(--text-muted)] hover:bg-[var(--bg-hover)]">
                      cancel
                    </button>
                  </div>
                </div>
              ) : (
                canonical.length > 1 && editing !== id && (
                  <button
                    onClick={() => setMergeSource(id)}
                    className="mt-0.5 ml-4 inline-flex items-center gap-0.5 text-[10px]
                               text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                  >
                    merge into another <ChevronRight size={9} />
                  </button>
                )
              )}
            </div>
          );
        })}

        {canonical.length === 0 && (
          <div className="px-3 py-4 text-center text-xs text-[var(--text-muted)]">
            <User size={16} className="mx-auto mb-1 opacity-50" />
            No speakers detected
          </div>
        )}
      </div>

      {mergedAway.length > 0 && (
        <div className="px-3 py-1.5 border-t border-[var(--border-subtle)] text-[10px] text-[var(--text-muted)]">
          {mergedAway.length} id{mergedAway.length === 1 ? '' : 's'} merged — click one above to split it back out
        </div>
      )}
    </div>
  );
}
