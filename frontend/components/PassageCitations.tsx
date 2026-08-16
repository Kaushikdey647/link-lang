"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Passage } from "@/lib/api";

interface Props {
  passages: Passage[];
}

export default function PassageCitations({ passages }: Props) {
  const [open, setOpen] = useState(false);

  if (!passages.length) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {passages.length} source{passages.length !== 1 ? "s" : ""}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex flex-col gap-2">
              {passages.map((p, i) => (
                <div
                  key={p.passage_id + i}
                  className="rounded-lg border border-border bg-muted/50 p-3 text-xs"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-muted-foreground">
                      [{i + 1}]
                    </span>
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                      {p.chunk_type}
                    </Badge>
                    {p.is_selected && (
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        ★
                      </Badge>
                    )}
                    <span className="font-mono text-muted-foreground text-[10px]">
                      {p.passage_id}
                    </span>
                  </div>
                  <p className="leading-relaxed text-foreground/80">{p.text}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
