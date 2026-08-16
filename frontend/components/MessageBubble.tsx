"use client";

import { motion } from "framer-motion";
import { Mic } from "lucide-react";
import PassageCitations from "./PassageCitations";
import type { Passage } from "@/lib/api";

export type Role = "user" | "assistant" | "error";

export interface Message {
  id: string;
  role: Role;
  text: string;
  transcript?: string; // set when role=user and came from voice
  passages?: Passage[];
}

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const isError = message.role === "error";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : isError
            ? "bg-destructive/10 text-destructive border border-destructive/20 rounded-bl-sm"
            : "bg-muted text-foreground rounded-bl-sm"
        }`}
      >
        {message.transcript && (
          <div className="flex items-center gap-1.5 mb-1.5 opacity-70 text-xs">
            <Mic size={10} />
            <span className="italic">{message.transcript}</span>
          </div>
        )}
        <p className="whitespace-pre-wrap">{message.text}</p>
        {message.passages && message.passages.length > 0 && (
          <PassageCitations passages={message.passages} />
        )}
      </div>
    </motion.div>
  );
}

export function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex justify-start"
    >
      <div className="bg-muted rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1">
        {[0, 0.15, 0.3].map((delay, i) => (
          <motion.span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60"
            animate={{ y: [0, -4, 0] }}
            transition={{ repeat: Infinity, duration: 0.6, delay }}
          />
        ))}
      </div>
    </motion.div>
  );
}
