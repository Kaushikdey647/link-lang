"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export const LANGUAGES: Record<string, string> = {
  "Hindi (हिंदी)": "hi",
  "Bengali (বাংলা)": "bn",
  "Gujarati (ગુજરાતી)": "gu",
  "Kannada (ಕನ್ನಡ)": "kn",
  "Malayalam (മലയാളം)": "ml",
  "Marathi (मराठी)": "mr",
  "Nepali (नेपाली)": "ne",
  "Odia (ଓଡ଼ିଆ)": "or",
  "Punjabi (ਪੰਜਾਬੀ)": "pa",
  "Sanskrit (संस्कृतम्)": "sa",
  "Tamil (தமிழ்)": "ta",
  "Telugu (తెలుగు)": "te",
  "Urdu (اردو)": "ur",
  "Assamese (অসমীয়া)": "as",
};

interface Props {
  value: string;
  onChange: (lang: string) => void;
  disabled?: boolean;
}

export default function LangSelector({ value, onChange, disabled }: Props) {
  const displayName =
    Object.entries(LANGUAGES).find(([, code]) => code === value)?.[0] ??
    "Hindi (हिंदी)";

  return (
    <Select
      value={value}
      onValueChange={(v) => v && onChange(v)}
      disabled={disabled}
    >
      <SelectTrigger className="w-44 text-xs text-white/60 bg-white/5 border-white/10 rounded-full px-4 hover:bg-white/10 transition-colors">
        <SelectValue>{displayName}</SelectValue>
      </SelectTrigger>
      <SelectContent className="max-h-72 bg-[#1a1a1a] border-white/10 text-white/70">
        {Object.entries(LANGUAGES).map(([label, code]) => (
          <SelectItem key={code} value={code} className="text-xs text-white/70 focus:bg-white/10 focus:text-white">
            {label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
