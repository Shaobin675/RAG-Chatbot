import { useState } from "react";
import { sendChatMessage } from "./api/chatApi";
import { uploadDocument } from "./api/uploadApi";

function App() {
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend() {
    setLoading(true);
    try {
      const response = await sendChatMessage({
        message,
        namespace: "default",
      });
      setAnswer(response.answer);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.length) return;

    await uploadDocument(
      e.target.files[0],
      "default",
      "00000000-0000-0000-0000-000000000001"
    );

    alert("Document uploaded. Indexing in progress.");
  }

  return (
    <div style={{ padding: 24 }}>
      <h2>RAG Chat</h2>

      <input type="file" onChange={handleUpload} />

      <div style={{ marginTop: 16 }}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
          cols={60}
        />
      </div>

      <button onClick={handleSend} disabled={loading}>
        {loading ? "Thinking..." : "Send"}
      </button>

      {answer && (
        <div style={{ marginTop: 24 }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
