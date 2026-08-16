"use client";

import { useRef, useState } from "react";
import { SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function TextInput({ onSend, disabled, placeholder }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset height
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    // Auto-grow
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  return (
    <div className="flex items-end gap-2 bg-white/5 rounded-2xl border border-white/10 px-4 py-3">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder ?? "Type your question…"}
        rows={1}
        className="flex-1 resize-none bg-transparent text-sm text-white/80 outline-none placeholder:text-white/25 leading-relaxed py-0.5 max-h-28"
      />
      <Button
        size="icon"
        variant="ghost"
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="shrink-0 h-7 w-7 rounded-xl text-white/40 hover:text-white/80 hover:bg-white/10"
      >
        <SendHorizontal size={14} />
      </Button>
    </div>
  );
}
