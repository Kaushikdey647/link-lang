"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Passage } from "@/lib/api";

interface Props {
  transcript?: string;
  answer: string;
  passages: Passage[];
  totalMs?: number;
}

function fmtDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export default function AnswerDisplay({ transcript, answer, passages, totalMs }: Props) {
  const [showSources, setShowSources] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full max-w-xl mx-auto flex flex-col gap-4 px-2"
    >
      {/* Transcript */}
      {transcript && (
        <p className="text-white/40 text-sm text-center tracking-wide">
          &ldquo;{transcript}&rdquo;
        </p>
      )}

      {/* Answer */}
      <div className="rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm px-6 py-5">
        <p className="text-white/90 text-base leading-relaxed whitespace-pre-wrap">{answer}</p>
      </div>

      {/* Timing + sources toggle */}
      {(totalMs !== undefined || passages.length > 0) && (
        <div>
          <div className="flex items-center justify-center gap-3">
            {totalMs !== undefined && (
              <span className="text-xs text-white/25 font-mono tracking-wide">
                {fmtDuration(totalMs)}
              </span>
            )}
            {passages.length > 0 && (
              <button
                onClick={() => setShowSources((s) => !s)}
                className="flex items-center gap-1.5 text-xs text-white/30 hover:text-white/60 transition-colors"
              >
                {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                {passages.length} source{passages.length !== 1 ? "s" : ""}
              </button>
            )}
          </div>

          {showSources && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden mt-3 flex flex-col gap-2"
            >
              {passages.map((p, i) => (
                <div
                  key={p.passage_id + i}
                  className="rounded-xl border border-white/8 bg-white/3 px-4 py-3 text-xs text-white/50 leading-relaxed"
                >
                  <span className="text-white/25 font-mono mr-2">[{i + 1}]</span>
                  {p.is_selected && <span className="mr-2 text-white/40">★</span>}
                  {p.text}
                </div>
              ))}
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
}
