"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Owns MediaRecorder + Web Speech API (live transcript only).
 * Language detection happens server-side via saaras:v3 — no lang param needed.
 * Web Speech API uses browser locale as a best-effort hint for live captions.
 */
export function useVoice(onResult: (blob: Blob) => void) {
  const [liveTranscript, setLiveTranscript] = useState("");

  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const mrRef     = useRef<MediaRecorder | null>(null);
  const srRef     = useRef<unknown>(null);
  const chunksRef = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    mrRef.current = mr;
    chunksRef.current = [];

    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      onResultRef.current(new Blob(chunksRef.current, { type: mr.mimeType }));
    };
    mr.start(100);
    setLiveTranscript("");

    // Live captions via Web Speech API — best-effort, browser decides language
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const w = window as any;
    const SR = w.SpeechRecognition ?? w.webkitSpeechRecognition;
    if (SR) {
      const sr = new SR();
      sr.interimResults = true;
      sr.continuous = true;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      sr.onresult = (e: any) => {
        let text = "";
        for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
        setLiveTranscript(text);
      };
      sr.onerror = () => {};
      sr.start();
      srRef.current = sr;
    }
  }, []);

  const stop = useCallback(() => {
    mrRef.current?.stop();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (srRef.current as any)?.stop();
    srRef.current = null;
    setLiveTranscript("");
  }, []);

  return { liveTranscript, start, stop };
}
