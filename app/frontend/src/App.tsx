// app/frontend/src/App.tsx
import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type Message = { role: "user" | "assistant"; text: string };

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const send = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await invoke<string>("invoke_generate", { prompt: input });
      const assistantMsg: Message = { role: "assistant", text: response };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      const assistantMsg: Message = { role: "assistant", text: "Error calling backend." };
      setMessages((prev) => [...prev, assistantMsg]);
    }

    setInput("");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", padding: 16 }}>
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
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type a message..."
        />
        <button onClick={send}>Send</button>
      </div>
    </div>
  );
}

export default App;
