"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square } from "lucide-react";

interface Props {
  isRecording: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

/**
 * Purely controlled — no internal recording state.
 * Parent drives isRecording; this component just renders and calls onToggle.
 */
export default function MicButton({ isRecording, onToggle, disabled }: Props) {
  return (
    <div className="relative flex items-center justify-center">
      {/* Pulse rings while recording */}
      <AnimatePresence>
        {isRecording && (
          <>
            {[0, 1].map((i) => (
              <motion.div
                key={i}
                className="absolute rounded-full border border-white/20"
                initial={{ width: 88, height: 88, opacity: 0.55 }}
                animate={{ width: 88 + (i + 1) * 42, height: 88 + (i + 1) * 42, opacity: 0 }}
                transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.55, ease: "easeOut" }}
              />
            ))}
          </>
        )}
      </AnimatePresence>

      <motion.button
        whileTap={{ scale: 0.92 }}
        onClick={onToggle}
        disabled={disabled}
        aria-label={isRecording ? "Stop recording" : "Start recording"}
        className={`
          relative z-10 w-[88px] h-[88px] rounded-full
          flex items-center justify-center
          transition-colors duration-300 shadow-2xl
          disabled:opacity-30 disabled:cursor-not-allowed
          ${isRecording
            ? "bg-white/20 backdrop-blur-md border border-white/30"
            : "bg-white/10 backdrop-blur-md border border-white/15 hover:bg-white/15"
          }
        `}
      >
        <motion.div
          animate={isRecording ? { scale: [1, 1.1, 1] } : { scale: 1 }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        >
          {isRecording
            ? <Square size={24} className="text-white" fill="white" />
            : <Mic   size={26} className="text-white" />
          }
        </motion.div>
      </motion.button>
    </div>
  );
}
