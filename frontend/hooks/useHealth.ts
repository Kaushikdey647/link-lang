"use client";

import { useEffect, useState } from "react";

type HealthState = "checking" | "ready" | "unreachable";

export function useHealth(pollMs = 8000): HealthState {
  const [state, setState] = useState<HealthState>("checking");

  useEffect(() => {
    let mounted = true;

    const check = async () => {
      try {
        const res = await fetch("/api/health", {
          signal: AbortSignal.timeout(3000),
          cache: "no-store",
        });
        if (mounted) {
          if (res.ok) {
            const body = await res.json().catch(() => ({}));
            setState(body.status === "ok" ? "ready" : "unreachable");
          } else {
            setState("unreachable");
          }
        }
      } catch {
        if (mounted) setState("unreachable");
      }
    };

    check();
    const id = setInterval(check, pollMs);
    return () => { mounted = false; clearInterval(id); };
  }, [pollMs]);

  return state;
}
