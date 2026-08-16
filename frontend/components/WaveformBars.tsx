"use client";

import { motion } from "framer-motion";

interface Props {
  active?: boolean;
  bars?: number;
  className?: string;
}

const HEIGHTS = [0.4, 0.7, 1, 0.6, 0.9, 0.5, 0.8, 0.4, 0.75, 0.55, 0.9, 1, 0.6, 0.45, 0.8];

export default function WaveformBars({ active = false, bars = 15, className = "" }: Props) {
  return (
    <div className={`flex items-center justify-center gap-[3px] ${className}`}>
      {HEIGHTS.slice(0, bars).map((h, i) => (
        <motion.div
          key={i}
          className="w-[3px] rounded-full bg-white/70"
          style={{ originY: 1 }}
          animate={
            active
              ? {
                  scaleY: [h, h * 0.4 + Math.random() * 0.6, h],
                  opacity: [0.5, 1, 0.5],
                }
              : { scaleY: h, opacity: 0.25 }
          }
          transition={
            active
              ? {
                  duration: 0.5 + (i % 4) * 0.1,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: i * 0.04,
                }
              : { duration: 0.3 }
          }
          initial={{ scaleY: h, height: 32 }}
        />
      ))}
    </div>
  );
}
