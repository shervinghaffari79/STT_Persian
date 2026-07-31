import React, { useState, useRef, useEffect } from 'react';
import {
  MessageSquare, Send, Plus, History, Trash2,
  Bot, User, Copy, Check, ChevronRight,
  Sparkles, Clock, Search, Square, Pencil, X
} from 'lucide-react';
import { ChatMessage, ChatSession, TranscriptionResult } from '../types';
import { streamChatCompletion, generateTitle } from '../services/localChat';
import { copyText } from '../utils/clipboard';
import Markdown from './Markdown';

interface ChatPanelProps {
  transcription: TranscriptionResult | null;
  externalMessage?: string;
  onExternalMessageUsed: () => void;
  /** Seek the audio/transcript to a point cited in an AI response, e.g.
   * "[04:12]" -- see Markdown.tsx's citation handling. Optional so this panel
   * still works (citations just render inert) if no player is wired up. */
  onSeek?: (time: number) => void;
}

type RightPanelView = 'chat' | 'history';

function generateId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function formatTimestamp(t: number): string {
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60).toString().padStart(2, '0');
  const s = Math.floor(t % 60).toString().padStart(2, '0');
  return h > 0 ? `${h}:${m}:${s}` : `${m}:${s}`;
}

// Per-segment timestamps so the model can cite a specific moment back at the
// user -- transcription.rawText (used for copy/export) deliberately omits
// these to stay a clean plain-text transcript, so this is a separate,
// chat-only view of the same segments. chat.py's system prompt instructs the
// model to cite in exactly this [MM:SS] bracket form; Markdown.tsx's
// citation regex is what makes a cited timestamp clickable in the reply.
function buildChatTranscript(transcription: TranscriptionResult): string {
  return transcription.segments
    .map(seg => `[${seg.speaker} ${formatTimestamp(seg.start)}]: ${seg.text}`)
    .join('\n\n');
}

const SUGGESTED_PROMPTS = [
  'نکات کلیدی مطرح‌شده را خلاصه کن',
  'گویندگان اصلی چه کسانی بودند و نقش آن‌ها چه بود؟',
  'چه اقدامات عملی مشخص شد؟',
  'احساس کلی مکالمه چگونه بود؟',
  'موضوعات اصلی را به صورت نقطه‌ای فهرست کن',
  'چه تصمیماتی در این جلسه گرفته شد؟',
];

export default function ChatPanel({
  transcription,
  externalMessage,
  onExternalMessageUsed,
  onSeek,
}: ChatPanelProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [view, setView] = useState<RightPanelView>('chat');
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const [copyErrorMsgId, setCopyErrorMsgId] = useState<string | null>(null);
  const [historySearch, setHistorySearch] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const activeSession = sessions.find(s => s.id === activeSessionId) || null;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);

  // Handle external messages (from transcript panel)
  useEffect(() => {
    if (externalMessage) {
      setInput(externalMessage);
      setView('chat');
      onExternalMessageUsed();
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [externalMessage, onExternalMessageUsed]);

  const createNewSession = () => {
    const newSession: ChatSession = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      transcriptionId: transcription?.id,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setView('chat');
  };

  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId));
    if (activeSessionId === sessionId) {
      const remaining = sessions.filter(s => s.id !== sessionId);
      setActiveSessionId(remaining[0]?.id || null);
    }
  };

  // Stream an assistant reply for `history` into a fresh AI message on `sessionId`.
  const generateReply = async (
    sessionId: string,
    history: ChatMessage[],
    makeTitleFrom?: string,
  ) => {
    const aiMsgId = generateId();
    const aiMsg: ChatMessage = {
      id: aiMsgId, role: 'assistant', content: '', timestamp: new Date(), isStreaming: true,
    };
    setSessions(prev => prev.map(s =>
      s.id === sessionId ? { ...s, messages: [...s.messages, aiMsg], updatedAt: new Date() } : s));

    setIsStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    await streamChatCompletion(
      history,
      transcription ? buildChatTranscript(transcription) : '',
      (token) => {
        setSessions(prev => prev.map(s =>
          s.id === sessionId
            ? { ...s, messages: s.messages.map(m => m.id === aiMsgId ? { ...m, content: m.content + token } : m) }
            : s));
      },
      async () => {
        setSessions(prev => prev.map(s =>
          s.id === sessionId
            ? { ...s, messages: s.messages.map(m => m.id === aiMsgId ? { ...m, isStreaming: false } : m) }
            : s));
        setIsStreaming(false);
        abortRef.current = null;
        if (makeTitleFrom) {
          const title = await generateTitle(makeTitleFrom);
          setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title } : s));
        }
      },
      (err) => {
        setSessions(prev => prev.map(s =>
          s.id === sessionId
            ? { ...s, messages: s.messages.map(m => m.id === aiMsgId ? { ...m, content: `⚠️ ${err}`, isStreaming: false } : m) }
            : s));
        setIsStreaming(false);
        abortRef.current = null;
      },
      controller.signal,
    );
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    let sessionId = activeSessionId;
    let isFirst = false;
    if (!sessionId) {
      const s: ChatSession = {
        id: generateId(), title: 'New Chat', messages: [],
        transcriptionId: transcription?.id, createdAt: new Date(), updatedAt: new Date(),
      };
      setSessions(prev => [s, ...prev]);
      sessionId = s.id;
      setActiveSessionId(sessionId);
      isFirst = true;
    } else {
      isFirst = (sessions.find(s => s.id === sessionId)?.messages.length || 0) === 0;
    }

    const userMsg: ChatMessage = { id: generateId(), role: 'user', content: text, timestamp: new Date() };
    setSessions(prev => prev.map(s =>
      s.id === sessionId ? { ...s, messages: [...s.messages, userMsg], updatedAt: new Date() } : s));

    const history = [...(sessions.find(s => s.id === sessionId)?.messages || []), userMsg];
    await generateReply(sessionId, history, isFirst ? text : undefined);
  };

  // Stop an in-progress generation.
  const handleStop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  };

  // Edit a previously-sent user message: truncate everything after it, then regenerate.
  const startEdit = (msg: ChatMessage) => {
    setEditingId(msg.id);
    setEditText(msg.content);
  };

  const saveEdit = async () => {
    const text = editText.trim();
    const sessionId = activeSessionId;
    if (!text || !sessionId || isStreaming) { setEditingId(null); return; }
    const session = sessions.find(s => s.id === sessionId);
    if (!session) { setEditingId(null); return; }
    const idx = session.messages.findIndex(m => m.id === editingId);
    if (idx < 0) { setEditingId(null); return; }

    // keep messages before the edited one, replace it with the new text
    const kept = session.messages.slice(0, idx);
    const editedMsg: ChatMessage = { ...session.messages[idx], content: text, timestamp: new Date() };
    const newMessages = [...kept, editedMsg];
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: newMessages } : s));
    setEditingId(null);
    setEditText('');
    await generateReply(sessionId, newMessages, kept.length === 0 ? text : undefined);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyMessage = async (id: string, content: string) => {
    const ok = await copyText(content);
    if (ok) {
      setCopiedMsgId(id);
      setTimeout(() => setCopiedMsgId(null), 2000);
    } else {
      setCopyErrorMsgId(id);
      setTimeout(() => setCopyErrorMsgId(null), 2500);
    }
  };

  const formatTime = (d: Date) => {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (d: Date) => {
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 86400000) return 'Today';
    if (diff < 172800000) return 'Yesterday';
    return d.toLocaleDateString();
  };

  const filteredSessions = sessions.filter(s =>
    !historySearch || s.title.toLowerCase().includes(historySearch.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-[var(--bg-base)]">
      {/* Header */}
      <div className="panel-header px-4 py-3 flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-emerald-500/20 flex items-center justify-center">
          <MessageSquare size={14} className="text-emerald-400" />
        </div>
        <span className="text-sm font-semibold text-[var(--text-primary)] flex-1">AI Analysis</span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setView('history')}
            className={`w-7 h-7 flex items-center justify-center rounded-lg transition-colors ${
              view === 'history' ? 'bg-indigo-500/20 text-indigo-300' : 'hover:bg-[var(--bg-active)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
            title="Chat history"
          >
            <History size={14} />
          </button>
          <button
            onClick={createNewSession}
            className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-[var(--bg-active)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="New chat"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      {/* History View */}
      {view === 'history' ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-[var(--border-shell)]">
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
              <input
                value={historySearch}
                onChange={e => setHistorySearch(e.target.value)}
                placeholder="Search history..."
                className="w-full bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-lg pl-7 pr-3 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:border-indigo-500/50 transition-colors"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            {filteredSessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <History size={24} className="text-[var(--text-muted)] mb-2" />
                <p className="text-sm text-[var(--text-muted)]">No chat history yet</p>
                <button
                  onClick={createNewSession}
                  className="mt-3 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                  Start a new chat
                </button>
              </div>
            ) : (
              filteredSessions.map(session => (
                <div
                  key={session.id}
                  className={`group flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-all ${
                    session.id === activeSessionId
                      ? 'bg-indigo-500/15 border border-indigo-500/30'
                      : 'hover:bg-[var(--bg-active)] border border-transparent'
                  }`}
                  onClick={() => { setActiveSessionId(session.id); setView('chat'); }}
                >
                  <div className="w-7 h-7 rounded-lg bg-[var(--bg-inset)] flex items-center justify-center flex-shrink-0">
                    <MessageSquare size={12} className="text-[var(--text-muted)]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-[var(--text-primary)] truncate">{session.title}</p>
                    <p className="text-[10px] text-[var(--text-muted)] flex items-center gap-1">
                      <Clock size={9} />
                      {formatDate(session.updatedAt)} · {session.messages.length} msgs
                    </p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteSession(session.id); }}
                    className="opacity-0 group-hover:opacity-100 w-6 h-6 flex items-center justify-center rounded-md hover:bg-red-500/20 text-[var(--text-muted)] hover:text-red-400 transition-all"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <>
          {/* Chat View */}
          <div className="flex-1 overflow-y-auto">
            {!activeSession || activeSession.messages.length === 0 ? (
              <div className="p-4 space-y-5">
                {/* Welcome */}
                <div className="text-center py-6">
                  <div className="w-14 h-14 bg-gradient-to-br from-emerald-500/20 to-indigo-500/20 rounded-2xl flex items-center justify-center mx-auto mb-3 border border-emerald-500/20">
                    <Sparkles size={24} className="text-emerald-400" />
                  </div>
                  <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                    {transcription ? 'Analyze Your Transcript' : 'AI Chat Assistant'}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)] max-w-[200px] mx-auto">
                    {transcription
                      ? 'Ask anything about your transcribed audio'
                      : 'Transcribe audio first for context-aware analysis'
                    }
                  </p>
                </div>

                {/* Suggested prompts */}
                {transcription && (
                  <div className="space-y-2">
                    <p className="text-[11px] text-[var(--text-muted)] font-medium uppercase tracking-wide">Suggested</p>
                    <div className="space-y-1.5">
                      {SUGGESTED_PROMPTS.map(prompt => (
                        <button
                          key={prompt}
                          onClick={() => {
                            setInput(prompt);
                            textareaRef.current?.focus();
                          }}
                          className="w-full text-left px-3 py-2 rounded-xl bg-[var(--bg-inset)] border border-[var(--border-shell)] hover:border-indigo-500/30 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all flex items-center gap-2"
                        >
                          <ChevronRight size={12} className="text-indigo-400 flex-shrink-0" />
                          {prompt}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 space-y-4">
                {activeSession.messages.map((msg) => (
                  <div key={msg.id} className={`fade-in flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                    {/* Avatar */}
                    <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                      msg.role === 'user'
                        ? 'bg-indigo-500/20'
                        : 'bg-emerald-500/20'
                    }`}>
                      {msg.role === 'user'
                        ? <User size={13} className="text-indigo-300" />
                        : <Bot size={13} className="text-emerald-300" />
                      }
                    </div>

                    {/* Bubble */}
                    <div className={`group flex-1 max-w-[85%] ${msg.role === 'user' ? 'flex flex-col items-end' : ''} min-w-0`}>
                      {editingId === msg.id ? (
                        /* inline edit of a user question */
                        <div className="w-full bg-[var(--bg-inset)] border border-indigo-500/50 rounded-2xl px-3 py-2">
                          <textarea
                            value={editText}
                            onChange={e => setEditText(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveEdit(); }
                              if (e.key === 'Escape') { setEditingId(null); }
                            }}
                            dir="auto"
                            autoFocus
                            className="w-full bg-transparent text-sm text-[var(--text-primary)] resize-none outline-none text-right leading-relaxed"
                            rows={2}
                          />
                          <div className="flex items-center justify-end gap-2 mt-1.5">
                            <button onClick={() => setEditingId(null)}
                              className="text-[11px] px-2 py-1 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-active)] flex items-center gap-1">
                              <X size={11} /> لغو
                            </button>
                            <button onClick={saveEdit} disabled={!editText.trim()}
                              className="text-[11px] px-2.5 py-1 rounded-lg bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 text-white flex items-center gap-1">
                              <Check size={11} /> ذخیره و ارسال
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className={`rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed break-words ${
                          msg.role === 'user'
                            ? 'chat-bubble-user text-[var(--text-primary)] rounded-tr-sm'
                            : 'chat-bubble-ai text-[var(--text-primary)] rounded-tl-sm'
                        }`}>
                          {msg.isStreaming && msg.content === '' ? (
                            <div className="flex items-center gap-1 py-1">
                              {[0, 1, 2].map(i => (
                                <div key={i} className="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce"
                                  style={{ animationDelay: `${i * 0.15}s` }} />
                              ))}
                            </div>
                          ) : msg.role === 'assistant' ? (
                            <div className={msg.isStreaming ? 'typing-cursor' : ''}>
                              <Markdown content={msg.content} onCite={onSeek} />
                            </div>
                          ) : (
                            <div className="whitespace-pre-wrap text-right" dir="auto">{msg.content}</div>
                          )}
                        </div>
                      )}

                      {/* Actions */}
                      {editingId !== msg.id && (
                        <div className={`flex items-center gap-2 mt-1 opacity-0 group-hover:opacity-100 transition-opacity ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                          <span className="text-[10px] text-[var(--text-muted)]">{formatTime(msg.timestamp)}</span>
                          {msg.role === 'user' && !isStreaming && (
                            <button onClick={() => startEdit(msg)}
                              className="text-[var(--text-muted)] hover:text-indigo-300 transition-colors" title="ویرایش سوال">
                              <Pencil size={11} />
                            </button>
                          )}
                          {msg.role === 'assistant' && !msg.isStreaming && (
                            <button onClick={() => copyMessage(msg.id, msg.content)}
                              className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
                              title={copyErrorMsgId === msg.id ? 'کپی ناموفق بود -- کلیپ‌بورد در دسترس نیست' : 'کپی'}>
                              {copiedMsgId === msg.id
                                ? <Check size={11} className="text-green-400" />
                                : copyErrorMsgId === msg.id
                                ? <X size={11} className="text-red-400" />
                                : <Copy size={11} />}
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-3 border-t border-[var(--border-shell)]">
            {/* Transcript context indicator */}
            {transcription && (
              <div className="flex items-center gap-1.5 mb-2 px-1">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <span className="text-[10px] text-[var(--text-muted)]">
                  Context: <span className="text-green-400">{transcription.fileName}</span>
                </span>
              </div>
            )}

            <div className="flex items-end gap-2 bg-[var(--bg-inset)] border border-[var(--border-subtle)] rounded-2xl px-3 py-2 focus-within:border-indigo-500/50 transition-colors">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={transcription ? 'سوال خود را بپرسید...' : 'یک مکالمه شروع کنید...'}
                dir="rtl"
                className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] resize-none max-h-32 min-h-[36px] leading-relaxed text-right"
                rows={1}
                style={{ height: 'auto' }}
                onInput={e => {
                  const el = e.currentTarget;
                  el.style.height = 'auto';
                  el.style.height = Math.min(el.scrollHeight, 128) + 'px';
                }}
              />
              {isStreaming ? (
                <button
                  onClick={handleStop}
                  title="توقف تولید"
                  className="w-8 h-8 bg-red-500/90 hover:bg-red-600 rounded-xl flex items-center justify-center transition-all flex-shrink-0"
                >
                  <Square size={12} className="text-white" fill="white" />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="w-8 h-8 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-all flex-shrink-0"
                >
                  <Send size={14} className="text-white" />
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
