import React, { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [papers, setPapers] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [showPapers, setShowPapers] = useState(false);
  const [lightbox, setLightbox] = useState(null);

  const loadPapers = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/papers`);
      const data = await res.json();
      setPapers(data.papers || []);
    } catch (err) {
      setError("Failed to load papers.");
    }
  };

  useEffect(() => {
    loadPapers();
  }, []);

  const uploadPaper = async (file) => {
    if (!file) return;
    setUploading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/api/papers/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Upload failed.");
      }
      const data = await res.json();
      setPapers(data.papers || []);
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const deletePaper = async (name) => {
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/papers/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Delete failed.");
      }
      const data = await res.json();
      setPapers(data.papers || []);
    } catch (err) {
      setError(err.message || "Delete failed.");
    }
  };

  const formatText = (text) => {
    if (!text) return "";
    return text
      .replace(/\r\n/g, "\n")
      .replace(/^\s*\*\s+/gm, "• ")
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/^\s*#+\s+/gm, "");
  };


  const submit = async (event) => {
    event.preventDefault();
    if (!message.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    const userMessage = message.trim();
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setMessage("");
    const historyPayload = [...messages, { role: "user", text: userMessage }];
    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          history: historyPayload,
          include_visuals: null,
          max_pages: null,
        }),
      });
      const data = await res.json();
      const visuals = data.visuals || [];
      const showVisuals = data.show_visuals;
      const visualNote =
        showVisuals && visuals.length === 0
          ? "\n\nNo relevant visuals were found for this question."
          : "";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `${data.answer || ""}${visualNote}`,
          visuals,
        },
      ]);
    } catch (err) {
      setError("Request failed. Check the backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="title-block">
          <h1>Research Paper Summarizer</h1>
          <div className="subtitle">RAG-powered paper Q&amp;A</div>
        </div>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => setShowPapers((prev) => !prev)}
        >
          {showPapers ? "Hide Papers" : "Manage Papers"}
        </button>
      </header>

      {error && <div className="error">{error}</div>}

      {showPapers && (
        <section className="papers-panel">
          <div>
            <h2>Papers</h2>
            <p>Upload PDF files and manage your collection.</p>
          </div>
          <div className="papers-actions">
            <label className="upload-btn">
              {uploading ? "Uploading..." : "Upload PDF"}
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => uploadPaper(e.target.files?.[0])}
                disabled={uploading}
              />
            </label>
            <button type="button" onClick={loadPapers} disabled={uploading}>
              Refresh
            </button>
          </div>
          <div className="papers-list">
            {papers.length === 0 && <div>No papers uploaded yet.</div>}
            {papers.map((paper) => (
              <div className="paper-row" key={paper}>
                <span>{paper}</span>
                <button type="button" onClick={() => deletePaper(paper)}>
                  Delete
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="chat">
        {messages.map((item, index) => (
          <div
            key={`${item.role}-${index}`}
            className={`bubble ${item.role}`}
          >
            {item.role === "assistant" &&
              item.visuals &&
              item.visuals.length > 0 && (
                <div className="visuals">
                  <div className="visual-grid">
                    {item.visuals.map((visual, visualIndex) => (
                      <div
                        className="visual-card"
                        key={`${visual.url}-${visualIndex}`}
                      >
                        <img
                          src={`${API_BASE}${visual.url}`}
                          alt={`Visual page ${visual.page}`}
                      onClick={() =>
                        setLightbox({
                          url: `${API_BASE}${visual.url}`,
                          page: visual.page,
                          captions: visual.captions || [],
                        })
                      }
                        />
                        <div className="caption">
                          <strong>Page {visual.page}</strong>
                          <ul>
                          {visual.captions.slice(0, 2).map((caption, idx) => (
                            <li key={`${visual.url}-${idx}`}>{caption}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            <div className="text">
              <pre>{formatText(item.text)}</pre>
            </div>
          </div>
        ))}
      </section>

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img src={lightbox.url} alt={`Visual page ${lightbox.page}`} />
            <div className="lightbox-caption">
              <strong>Page {lightbox.page}</strong>
              {lightbox.captions.length > 0 && (
                <ul>
                  {lightbox.captions.slice(0, 3).map((caption, idx) => (
                    <li key={`lightbox-${idx}`}>{caption}</li>
                  ))}
                </ul>
              )}
            </div>
            <button
              type="button"
              className="ghost-btn"
              onClick={() => setLightbox(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <footer className="composer">
        <form onSubmit={submit}>
          <textarea
            id="message"
            rows={3}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask anything about the papers..."
          />

          <div className="row">
            <button type="submit" disabled={loading}>
              {loading ? "Working..." : "Send"}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}

