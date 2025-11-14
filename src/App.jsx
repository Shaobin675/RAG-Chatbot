import React, { useState, useRef } from "react";
import "./index.css";

const IDLE_SECONDS = 180;

function App() {
  const [ws, setWs] = useState(null);
  const [username, setUsername] = useState("");
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [countdown, setCountdown] = useState(IDLE_SECONDS);

  const countdownInterval = useRef(null);
  const chatRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---------------- Scroll to bottom ----------------
  const scrollToBottom = () => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  };

  // ---------------- Add message ----------------
  const addMessage = (text, type = "user") => {
    setMessages((prev) => [...prev, { text, type }]);
    scrollToBottom();
    resetCountdown();
  };

  const addBotMessage = (text) => addMessage(text, "bot");

  // ---------------- Reset countdown ----------------
  const resetCountdown = () => {
    setCountdown(IDLE_SECONDS);
    if (countdownInterval.current) clearInterval(countdownInterval.current);
    countdownInterval.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(countdownInterval.current);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // ---------------- Connect WebSocket ----------------
  const connectWebSocket = () => {
    if (!username.trim()) return alert("Enter username");

    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/${username}`);
    setWs(socket);

    socket.onopen = () => addMessage(`Connected as ${username}`, "user");

    socket.onmessage = (event) => {
      const msg = event.data;

      // ---------------- File upload status ----------------
      if (msg.startsWith("📁 Received") || msg.startsWith("✅")) {
        addMessage(msg, "user");
      }
      // ---------------- File summary ----------------
      else if (msg.startsWith("📝")) {
        const summaryText = msg.replace(/^📝\s*/, "");
        const words = summaryText.split(" ");
        const truncated = words.slice(0, 250).join(" ");
        addBotMessage(truncated + (words.length > 250 ? "..." : ""));
      }
      // ---------------- Warnings / idle ----------------
      else if (msg.startsWith("⚠️")) {
        addMessage(msg, "warning");
      }
      // ---------------- Normal bot response ----------------
      else {
        addBotMessage(msg);
      }
    };

    socket.onclose = () => {
      addMessage("Disconnected from server", "warning");
      if (countdownInterval.current) clearInterval(countdownInterval.current);
    };
  };

  // ---------------- Send user message ----------------
  const sendMessage = () => {
    if (!userInput.trim() || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(userInput);
    addMessage(userInput, "user");
    setUserInput("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  // ---------------- File upload ----------------
  const allowedTypes = [
    "application/pdf",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ];

  const uploadFile = (file) => {
    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
      alert("Unsupported file type.");
      return;
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      alert("Connect first before uploading.");
      return;
    }

    addMessage(`Uploaded ${file.name} added to knowledge base.`, "user");

    const reader = new FileReader();
    reader.onload = () => {
      const base64Data = reader.result.split(",")[1];
      const payload = {
        type: "file_upload",
        filename: file.name,
        content_type: file.type,
        data: base64Data,
      };
      ws.send(JSON.stringify(payload));
      addBotMessage("Bot is typing...");
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="container">
      <h2>Multiuser Chat</h2>

      <div className="login">
        <input
          type="text"
          placeholder="Enter username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <button onClick={connectWebSocket}>Connect</button>
      </div>

      <div id="chatContainer">
        <div id="chat" ref={chatRef}>
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.type}`}>
              <img
                className="avatar"
                src={
                  msg.type === "bot"
                    ? "bot_avatar.png"
                    : msg.type === "user"
                    ? "user_avatar.png"
                    : ""
                }
                alt=""
              />
              <span className="message-text">{msg.text}</span>
            </div>
          ))}
        </div>

        <div id="progressContainer">
          <div
            id="progressBar"
            style={{ width: `${(countdown / IDLE_SECONDS) * 100}%` }}
          ></div>
        </div>

        <div className="inputArea">
          <input
            id="userInput"
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyUp={handleKeyPress}
          />
          <button onClick={sendMessage}>Send</button>
          <button onClick={() => fileInputRef.current.click()}>📎 Upload</button>
          <input
            type="file"
            style={{ display: "none" }}
            ref={fileInputRef}
            onChange={(e) => uploadFile(e.target.files[0])}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
