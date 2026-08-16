"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import MicButton from "@/components/MicButton";
import WaveformBars from "@/components/WaveformBars";
import AnswerDisplay from "@/components/AnswerDisplay";
import TextInput from "@/components/TextInput";
import { useVoice } from "@/hooks/useVoice";
import { useHealth } from "@/hooks/useHealth";
import { queryText, queryVoice } from "@/lib/api";
import type { Passage } from "@/lib/api";

// ── State machine ────────────────────────────────────────────────────────────
//
//   idle → recording → processing → answered
//                                 └→ error
//   (any) → recording  (tap mic again)
//
type UIState = "idle" | "recording" | "processing" | "answered" | "error";

interface Result {
  transcript?: string;
  detectedLang?: string;
  answer: string;
  passages: Passage[];
  totalMs: number;
}

const _fade = {
  initial:    { opacity: 0, y: 8 },
  animate:    { opacity: 1, y: 0 },
  exit:       { opacity: 0, y: -8 },
  transition: { duration: 0.2 },
};

// Human-readable language names for the detected-lang badge
const LANG_NAMES: Record<string, string> = {
  hi: "हिंदी", bn: "বাংলা", ta: "தமிழ்", te: "తెలుగు",
  kn: "ಕನ್ನಡ", ml: "മലയാളം", mr: "मराठी", gu: "ગુજરાતી",
  pa: "ਪੰਜਾਬੀ", or: "ଓଡ଼ିଆ", ne: "नेपाली", ur: "اردو",
  as: "অসমীয়া", sa: "संस्कृतम्", en: "English",
};

function fmtTime(s: number) {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [uiState,  setUiState]  = useState<UIState>("idle");
  const [result,   setResult]   = useState<Result | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [seconds,  setSeconds]  = useState(0);
  const [showText, setShowText] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Voice result callback ─────────────────────────────────────────────────

  const handleVoiceResult = useCallback(async (blob: Blob) => {
    setUiState("processing");
    const t0 = performance.now();
    try {
      const res = await queryVoice(blob);
      setResult({
        transcript: res.transcript,
        detectedLang: res.detected_lang,
        answer: res.answer,
        passages: res.passages,
        totalMs: performance.now() - t0,
      });
      setUiState("answered");
    } catch (e) {
      setErrorMsg(String(e));
      setUiState("error");
    }
  }, []);

  const voice = useVoice(handleVoiceResult);

  // ── Recording timer ───────────────────────────────────────────────────────

  useEffect(() => {
    if (uiState === "recording") {
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      if (uiState !== "processing") setSeconds(0);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [uiState]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleToggle = useCallback(async () => {
    if (uiState === "recording") {
      voice.stop();   // → mr.onstop → handleVoiceResult → "processing"
    } else {
      setResult(null);
      setErrorMsg("");
      setUiState("recording");
      try {
        await voice.start();
      } catch {
        setErrorMsg("Microphone access denied. Allow mic access and try again.");
        setUiState("error");
      }
    }
  }, [uiState, voice]);

  const handleText = useCallback(async (text: string) => {
    setShowText(false);
    setUiState("processing");
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await queryText(text);
      setResult({ answer: res.answer, passages: res.passages, totalMs: performance.now() - t0 });
      setUiState("answered");
    } catch (e) {
      setErrorMsg(String(e));
      setUiState("error");
    }
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────────

  const health       = useHealth();
  const isRecording  = uiState === "recording";
  const isProcessing = uiState === "processing";
  const backendReady = health === "ready";

  return (
    <div className="relative flex flex-col h-screen overflow-hidden bg-[#0d0d0d] text-white select-none">

      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="w-[500px] h-[500px] rounded-full bg-white/[0.02] blur-3xl" />
      </div>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="relative z-10 flex items-center justify-between px-6 pt-8 pb-2">
        <div className="flex flex-col gap-1">
          <span className="text-xs text-white/30 tracking-[0.2em] uppercase">Bhasha</span>
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-medium text-white/70 tracking-tight">Voice AI</span>
            <span
              title={health === "ready" ? "Backend ready" : health === "checking" ? "Connecting…" : "Backend unreachable"}
              className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${
                health === "ready"     ? "bg-white/50" :
                health === "checking" ? "bg-white/20 animate-pulse" :
                                        "bg-red-500/60"
              }`}
            />
          </div>
        </div>

        {/* Detected language badge — shows after a voice answer */}
        <AnimatePresence>
          {result?.detectedLang && (
            <motion.div
              key="lang-badge"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-xs text-white/30 bg-white/5 border border-white/10 rounded-full px-3 py-1 tracking-wide"
            >
              {LANG_NAMES[result.detectedLang] ?? result.detectedLang}
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* ── Main — state-specific center content ────────────────────────────── */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center gap-6 px-6 py-4 overflow-y-auto">
        <AnimatePresence mode="wait">

          {/* idle */}
          {uiState === "idle" && (
            <motion.div key="idle" {..._fade} className="flex flex-col items-center gap-3 text-center">
              <WaveformBars bars={11} active={false} className="mb-2" />
              <p className="text-white/25 text-sm tracking-wide">
                Speak in any of 14 Indian languages
              </p>
            </motion.div>
          )}

          {/* recording — live transcript */}
          {uiState === "recording" && (
            <motion.div key="recording" {..._fade} className="flex flex-col items-center gap-4 w-full max-w-sm px-4 text-center">
              <AnimatePresence mode="wait">
                {voice.liveTranscript ? (
                  <motion.p key="words" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="text-white/75 text-base leading-relaxed">
                    {voice.liveTranscript}
                  </motion.p>
                ) : (
                  <motion.p key="hint" initial={{ opacity: 0 }} animate={{ opacity: 0.35 }}
                    className="text-sm tracking-widest">
                    Listening…
                  </motion.p>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {/* processing */}
          {uiState === "processing" && (
            <motion.div key="processing" {..._fade} className="flex flex-col items-center gap-4">
              <WaveformBars bars={11} active={true} />
              <p className="text-white/40 text-sm tracking-widest">One moment…</p>
            </motion.div>
          )}

          {/* answered */}
          {uiState === "answered" && result && (
            <motion.div key="answered" {..._fade} className="w-full">
              <AnswerDisplay
                transcript={result.transcript}
                answer={result.answer}
                passages={result.passages}
                totalMs={result.totalMs}
              />
            </motion.div>
          )}

          {/* error */}
          {uiState === "error" && (
            <motion.div key="error" {..._fade}
              className="rounded-2xl bg-white/5 border border-white/10 px-6 py-5 max-w-sm text-center flex flex-col gap-2">
              <p className="text-white/50 text-sm">{errorMsg}</p>
              <p className="text-white/20 text-xs tracking-wide">
                {backendReady ? "Tap the mic to try again" : "Waiting for backend…"}
              </p>
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* ── Bottom controls ─────────────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col items-center gap-4 px-6 pb-10 pt-2">

        {/* Recording: timer + waveform */}
        <AnimatePresence>
          {isRecording && (
            <motion.div key="rec-chrome" {..._fade} className="flex flex-col items-center gap-2">
              <span className="text-xs text-white/35 font-mono tabular-nums tracking-widest">
                {fmtTime(seconds)}
              </span>
              <WaveformBars bars={15} active={true} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Mic button — hidden only during processing */}
        {!isProcessing && (
          <div className="flex flex-col items-center gap-1">
            <MicButton
              isRecording={isRecording}
              onToggle={handleToggle}
              disabled={!backendReady && !isRecording}
            />
            {health === "unreachable" && !isRecording && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="text-xs text-red-400/60 tracking-wide">
                Backend offline
              </motion.p>
            )}
            {health === "checking" && !isRecording && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="text-xs text-white/25 tracking-wide animate-pulse">
                Connecting…
              </motion.p>
            )}
          </div>
        )}

        {/* Text input — secondary */}
        <div className="w-full max-w-sm">
          {!showText ? (
            <button
              onClick={() => setShowText(true)}
              className="w-full text-center text-xs text-white/20 hover:text-white/40 transition-colors tracking-widest py-1"
            >
              type instead
            </button>
          ) : (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
              <TextInput
                onSend={handleText}
                disabled={isRecording || isProcessing}
                placeholder="Type in any Indian language…"
              />
            </motion.div>
          )}
        </div>

      </div>
    </div>
  );
}
