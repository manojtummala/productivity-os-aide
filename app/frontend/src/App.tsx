// app/frontend/src/App.tsx

import React, { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Message = { role: "user" | "assistant"; text: string };

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [isGenerating, setIsGenerating] = useState(false);

  const send = async () => {
    if (!input.trim() || isGenerating) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: input },
      { role: "assistant", text: "" },
    ]);

    setIsGenerating(true);
    setStatus("starting");

    try {
      await invoke("invoke_rag_generate_stream", { query: input });
    } catch (e) {
      console.error("invoke failed:", e);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", text: `Backend invoke error: ${String(e)}` },
      ]);
      setIsGenerating(false);
      setStatus("error");
    }

    setInput("");
  };

  const stop = async () => {
    try {
      await invoke("cancel_generation");
    } catch (e) {
      console.error("cancel failed:", e);
    }
  };

  useEffect(() => {
    const unlistenToken = listen<string>("llm-token", (event) => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant") return prev;
        return [...prev.slice(0, -1), { ...last, text: last.text + event.payload }];
      });
    });

    const unlistenStatus = listen<string>("rag-status", (event) => {
      setStatus(event.payload);
      if (event.payload === "generating") setIsGenerating(true);
      if (event.payload === "done") setIsGenerating(false);
    });

    const unlistenErr = listen<string>("rag-error", (event) => {
      setStatus("error");
      setIsGenerating(false);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", text: `Backend error: ${event.payload}` },
      ]);
    });

    const unlistenDone = listen("llm-done", () => {
      setStatus("done");
      setIsGenerating(false);
    });

    const unlistenStopped = listen("llm-stopped", () => {
      setStatus("stopped");
      setIsGenerating(false);
    });

    return () => {
      unlistenToken.then((f) => f());
      unlistenStatus.then((f) => f());
      unlistenErr.then((f) => f());
      unlistenDone.then((f) => f());
      unlistenStopped.then((f) => f());
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", padding: 16 }}>
      <div style={{ marginBottom: 8, fontSize: 12, opacity: 0.8 }}>
        Status: {status}
      </div>

      <div style={{ flex: 1, overflowY: "auto", marginBottom: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ margin: "4px 0" }}>
            <strong>{m.role === "user" ? "You" : "Assistant"}:</strong> {m.text}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          style={{ flex: 1, padding: 8 }}
          value={input}
          disabled={isGenerating}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type a message..."
        />
        <button onClick={send} disabled={isGenerating}>
          Send
        </button>
        <button onClick={stop} disabled={!isGenerating}>
          Stop
        </button>
      </div>
    </div>
  );
}

export default App;
