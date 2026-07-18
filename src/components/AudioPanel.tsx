import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Upload, Play, Pause, SkipBack, SkipForward,
  Volume2, VolumeX, Mic, FileAudio,
  Loader2, CheckCircle2, AlertCircle
} from 'lucide-react';
import { TranscriptionResult } from '../types';
import { transcribeLocal } from '../services/localAsr';

interface AudioPanelProps {
  onTranscriptionComplete: (result: TranscriptionResult) => void;
  onPartialTranscription?: (result: TranscriptionResult) => void;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  onAudioLoaded: (duration: number) => void;
  transcription: TranscriptionResult | null;
}

const SELECTED_LANG = 'fa';

type ProcessingStep = 'idle' | 'uploading' | 'processing' | 'fetching' | 'done' | 'error';

export default function AudioPanel({
  onTranscriptionComplete,
  onPartialTranscription,
  currentTime,
  onTimeUpdate,
  onAudioLoaded,
  transcription,
}: AudioPanelProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const progressFillRef = useRef<HTMLDivElement>(null);
  const timeDisplayRef = useRef<HTMLSpanElement>(null);
  const localTimeRef = useRef(0);
  const durationRef = useRef(0);
  const waveformDataRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [processingStep, setProcessingStep] = useState<ProcessingStep>('idle');
  const [processingMsg, setProcessingMsg] = useState('');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [waveformData, setWaveformData] = useState<number[]>([]);

  // Generate fake waveform data when file loaded
  const generateWaveform = useCallback((audioDuration: number) => {
    const bars = 80;
    const data = Array.from({ length: bars }, () => 0.2 + Math.random() * 0.8);
    waveformDataRef.current = data;
    durationRef.current = audioDuration;
    setWaveformData(data);
    setDuration(audioDuration);
    onAudioLoaded(audioDuration);
  }, [onAudioLoaded]);

  // RAF loop: drives canvas, progress bar, and time display without React re-renders
  useEffect(() => {
    if (waveformData.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const t = localTimeRef.current;
      const dur = durationRef.current;
      const bars = waveformDataRef.current;
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const barW = W / bars.length - 1;
      const progress = dur > 0 ? t / dur : 0;

      bars.forEach((val, i) => {
        const x = i * (barW + 1);
        const barH = val * H * 0.85;
        const y = (H - barH) / 2;
        ctx.fillStyle = (i / bars.length) <= progress ? '#6366f1' : '#2a2a2a';
        const radius = barW / 2;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, radius);
        ctx.fill();
      });

      // Update progress bar directly — no React state
      if (progressFillRef.current) {
        progressFillRef.current.style.width = `${dur > 0 ? (t / dur) * 100 : 0}%`;
      }

      // Update time display directly — no React state
      if (timeDisplayRef.current) {
        const m = Math.floor(t / 60).toString().padStart(2, '0');
        const s = Math.floor(t % 60).toString().padStart(2, '0');
        timeDisplayRef.current.textContent = `${m}:${s}`;
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [waveformData]);

  // Sync only genuine external seeks (from TranscriptPanel clicking a segment)
  // Ignore updates that originated from our own onTimeUpdate to break the feedback loop
  useEffect(() => {
    if (audioRef.current && Math.abs(currentTime - localTimeRef.current) > 0.5) {
      audioRef.current.currentTime = currentTime;
      localTimeRef.current = currentTime;
    }
  }, [currentTime]);

  const handleFileSelect = async (file: File) => {
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setProcessingStep('idle');
    setErrorMsg('');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('audio/') || file.type.startsWith('video/') || /\.(m4a|mp3|wav|ogg|flac|aac|opus|mp4)$/i.test(file.name))) {
      handleFileSelect(file);
    }
  };

  const handleTranscribe = async () => {
    if (!audioFile) return;
    setProcessingStep('uploading');
    setProcessingProgress(3);
    setErrorMsg('');

    try {
      setProcessingStep('processing');
      setProcessingMsg('Running local Whisper model…');

      const t0 = Date.now();
      const res = await transcribeLocal(
        audioFile,
        { diarize: true },
        (msg, pct) => {
          setProcessingMsg(msg);
          setProcessingProgress(pct);
        },
        // live partial transcript — render segments as they are generated
        (segments, speakers) => {
          onPartialTranscription?.({
            id: 'partial',
            fileName: audioFile.name,
            duration,
            language: SELECTED_LANG,
            createdAt: new Date(),
            segments,
            rawText: segments.map(s => `[${s.speaker}]: ${s.text}`).join('\n\n'),
            speakers,
            processingTime: (Date.now() - t0) / 1000,
          });
        },
      );

      const result: TranscriptionResult = {
        id: Date.now().toString(),
        fileName: audioFile.name,
        duration: res.duration || duration,
        language: res.language || SELECTED_LANG,
        createdAt: new Date(),
        segments: res.segments,
        rawText: res.rawText,
        speakers: res.speakers,
        processingTime: res.processingTime ?? (Date.now() - t0) / 1000,
      };

      setProcessingStep('done');
      setProcessingProgress(100);
      onTranscriptionComplete(result);
    } catch (err: any) {
      setProcessingStep('error');
      setErrorMsg(err.message || 'Transcription failed');
    }
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const skip = (secs: number) => {
    if (!audioRef.current) return;
    const newTime = Math.max(0, Math.min(durationRef.current, localTimeRef.current + secs));
    audioRef.current.currentTime = newTime;
    localTimeRef.current = newTime;
    onTimeUpdate(newTime);
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressRef.current || !audioRef.current) return;
    const rect = progressRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const newTime = pct * durationRef.current;
    audioRef.current.currentTime = newTime;
    localTimeRef.current = newTime;
    onTimeUpdate(newTime);
  };

  const formatTime = (t: number) => {
    const m = Math.floor(t / 60).toString().padStart(2, '0');
    const s = Math.floor(t % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const rates = [0.5, 0.75, 1, 1.25, 1.5, 2];

  return (
    <div className="flex flex-col h-full bg-[#0f0f0f]">
      {/* Header */}
      <div className="panel-header px-4 py-3 flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-indigo-500/20 flex items-center justify-center">
          <Mic size={14} className="text-indigo-400" />
        </div>
        <span className="text-sm font-semibold text-white">Audio Input</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Upload Zone */}
        <div
          className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-indigo-500 bg-indigo-500/10'
              : audioFile
              ? 'border-green-500/40 bg-green-500/5'
              : 'border-[#2a2a2a] hover:border-[#3a3a3a] hover:bg-white/[0.02]'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="audio/*,video/mp4,.m4a,.mp3,.wav,.ogg,.flac,.aac,.opus,.mp4"
            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
          />

          {audioFile ? (
            <div className="space-y-2">
              <div className="w-10 h-10 bg-green-500/20 rounded-full flex items-center justify-center mx-auto">
                <FileAudio size={20} className="text-green-400" />
              </div>
              <p className="text-sm font-medium text-white truncate max-w-[180px] mx-auto">{audioFile.name}</p>
              <p className="text-xs text-gray-500">{(audioFile.size / 1e6).toFixed(1)} MB</p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="w-10 h-10 bg-[#1e1e1e] rounded-full flex items-center justify-center mx-auto">
                <Upload size={20} className="text-gray-400" />
              </div>
              <p className="text-sm text-gray-300 font-medium">Drop audio or click to upload</p>
              <p className="text-xs text-gray-500">MP3, WAV, M4A, FLAC, OGG, AAC</p>
            </div>
          )}
        </div>

        {/* Audio Player */}
        {audioUrl && (
          <div className="bg-[#161616] rounded-xl p-4 space-y-3 border border-[#222]">
            <audio
              ref={audioRef}
              src={audioUrl}
              onTimeUpdate={(e) => {
                // Only update the ref — no React state, no parent callback.
                // This prevents re-rendering the entire app 4× per second during playback.
                localTimeRef.current = e.currentTarget.currentTime;
              }}
              onLoadedMetadata={(e) => generateWaveform(e.currentTarget.duration)}
              onEnded={() => setIsPlaying(false)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />

            {/* Waveform */}
            <div className="relative h-16">
              <canvas
                ref={canvasRef}
                width={220}
                height={64}
                className="w-full h-full"
              />
            </div>

            {/* Progress bar */}
            <div
              ref={progressRef}
              className="h-1.5 bg-[#2a2a2a] rounded-full cursor-pointer relative"
              onClick={handleProgressClick}
            >
              <div
                ref={progressFillRef}
                className="h-full bg-indigo-500 rounded-full transition-none relative"
                style={{ width: '0%' }}
              >
                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow -mr-1.5" />
              </div>
            </div>

            {/* Time */}
            <div className="flex justify-between text-xs text-gray-500">
              <span ref={timeDisplayRef}>00:00</span>
              <span>{formatTime(duration)}</span>
            </div>

            {/* Controls */}
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={() => skip(-10)}
                className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
              >
                <SkipBack size={16} />
              </button>
              <button
                onClick={togglePlay}
                className="w-10 h-10 bg-indigo-500 hover:bg-indigo-600 rounded-full flex items-center justify-center transition-colors"
              >
                {isPlaying ? <Pause size={18} fill="white" className="text-white" /> : <Play size={18} fill="white" className="text-white ml-0.5" />}
              </button>
              <button
                onClick={() => skip(10)}
                className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
              >
                <SkipForward size={16} />
              </button>
            </div>

            {/* Volume + Speed */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => {
                  const next = !isMuted;
                  setIsMuted(next);
                  if (audioRef.current) audioRef.current.muted = next;
                }}
                className="text-gray-400 hover:text-white transition-colors"
              >
                {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setVolume(v);
                  setIsMuted(false);
                  if (audioRef.current) {
                    audioRef.current.volume = v;
                    audioRef.current.muted = false;
                  }
                }}
                className="flex-1 accent-indigo-500 h-1"
              />
            </div>

            {/* Playback speed */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Speed:</span>
              <div className="flex gap-1 flex-wrap">
                {rates.map(r => (
                  <button
                    key={r}
                    onClick={() => {
                      setPlaybackRate(r);
                      if (audioRef.current) audioRef.current.playbackRate = r;
                    }}
                    className={`text-xs px-2 py-0.5 rounded-md transition-colors ${
                      playbackRate === r
                        ? 'bg-indigo-500 text-white'
                        : 'text-gray-400 hover:text-white bg-[#222] hover:bg-[#2a2a2a]'
                    }`}
                  >
                    {r}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Processing Status */}
        {processingStep !== 'idle' && processingStep !== 'done' && processingStep !== 'error' && (
          <div className="bg-[#161616] border border-[#222] rounded-xl p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Loader2 size={14} className="text-indigo-400 animate-spin" />
              <span className="text-xs text-gray-300 font-medium">Transcribing...</span>
            </div>
            <p className="text-xs text-gray-500">{processingMsg}</p>
            <div className="h-1 bg-[#2a2a2a] rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full progress-bar-fill"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
          </div>
        )}

        {processingStep === 'done' && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-3 flex items-center gap-2">
            <CheckCircle2 size={14} className="text-green-400" />
            <span className="text-xs text-green-300">Transcription complete!</span>
          </div>
        )}

        {processingStep === 'error' && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 space-y-1">
            <div className="flex items-center gap-2">
              <AlertCircle size={14} className="text-red-400" />
              <span className="text-xs text-red-300 font-medium">Error occurred</span>
            </div>
            <p className="text-xs text-red-400/70">{errorMsg}</p>
          </div>
        )}

        {/* Transcribe Button */}
        <div className="space-y-2">
          <button
            onClick={handleTranscribe}
            disabled={!audioFile || (processingStep !== 'idle' && processingStep !== 'error' && processingStep !== 'done')}
            className="w-full py-2.5 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl transition-all flex items-center justify-center gap-2"
          >
            {processingStep === 'uploading' || processingStep === 'processing' || processingStep === 'fetching' ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Mic size={14} />
                {processingStep === 'done' ? 'Re-Transcribe' : 'Transcribe Audio'}
              </>
            )}
          </button>

        </div>

        {/* Transcription Info */}
        {transcription && (
          <div className="bg-[#161616] border border-[#222] rounded-xl p-3 space-y-2">
            <p className="text-xs font-semibold text-gray-300">Last Transcription</p>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">File</span>
                <span className="text-gray-300 truncate max-w-[120px]">{transcription.fileName}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Duration</span>
                <span className="text-gray-300">{formatTime(transcription.duration)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Speakers</span>
                <span className="text-gray-300">{transcription.speakers.length} detected</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Language</span>
                <span className="text-gray-300">فارسی</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
