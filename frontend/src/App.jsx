import React, { useState, useEffect, useRef } from 'react';
import {
  Satellite,
  Radio,
  Cpu,
  Cloud,
  FileText,
  Upload,
  Trash2,
  Send,
  RefreshCw,
  Eye,
  Shield,
  CheckCircle2,
  AlertCircle,
  Maximize2,
  X,
  Key,
  Layers,
  Sliders,
  ChevronRight,
  Database
} from 'lucide-react';

export default function App() {
  // Application State
  const [messages, setMessages] = useState([]);
  const [queryInput, setQueryInput] = useState('');
  const [isTransmitting, setIsTransmitting] = useState(false);
  const [activeGraphStep, setActiveGraphStep] = useState(null);

  // Private Session Sandbox State
  const [attachedDoc, setAttachedDoc] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Model & System Config
  const [modelMode, setModelMode] = useState('offline'); // 'offline' | 'online'
  const [apiKey, setApiKey] = useState('');
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [isSyncing, setIsSyncing] = useState(false);

  // Visual Page Inspector & Lightbox
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorResults, setInspectorResults] = useState([]);
  const [lightboxImage, setLightboxImage] = useState(null);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Fetch initial system status
  useEffect(() => {
    fetchSystemStatus();
  }, []);

  // Auto-scroll chat on updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTransmitting]);

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (err) {
      console.warn('Status check warning:', err);
    }
  };

  // Re-sync Knowledge Corpus
  const handleSyncCorpus = async () => {
    setIsSyncing(true);
    try {
      const res = await fetch('/api/corpus/sync', { method: 'POST' });
      if (res.ok) {
        await fetchSystemStatus();
      }
    } catch (err) {
      console.error('Failed to sync corpus:', err);
    } finally {
      setIsSyncing(false);
    }
  };

  // Upload private flight document
  const handleFileUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only aerospace specification PDF documents are supported.');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/session/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Upload failed: ' + (await res.text()));
      }

      const data = await res.json();
      setAttachedDoc({
        sessionId: data.session_id,
        sessionRoot: data.session_root,
        documentName: data.document_name,
        totalPages: data.total_pages,
      });

      // System notification message
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'system',
          content: `🛰️ Attached private flight spec: **${data.document_name}** (${data.total_pages} pages). Isolated in encrypted runtime sandbox.`,
        },
      ]);
    } catch (err) {
      alert('Document attachment error: ' + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  // Detach and clean up private session
  const handleDetachDoc = async () => {
    if (!attachedDoc) return;
    try {
      await fetch('/api/session/detach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: attachedDoc.sessionId }),
      });
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: 'system',
          content: `🔒 Ephemeral sandbox for **${attachedDoc.documentName}** completely purged from disk.`,
        },
      ]);
      setAttachedDoc(null);
    } catch (err) {
      console.error('Detach error:', err);
    }
  };

  // Send Chat Query with Real-time SSE streaming
  const handleTransmit = async (queryText = queryInput) => {
    const q = queryText.trim();
    if (!q || isTransmitting) return;

    if (modelMode === 'online' && !apiKey) {
      setShowKeyModal(true);
      return;
    }

    const userMessageId = Date.now();
    const assistantMessageId = userMessageId + 1;

    // Append user message
    setMessages((prev) => [
      ...prev,
      { id: userMessageId, role: 'user', content: q },
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        citations: [],
        visual_items: [],
        retrieval_results: [],
        isStreaming: true,
      },
    ]);

    setQueryInput('');
    setIsTransmitting(true);
    setActiveGraphStep('understand');

    // Simulate graph progression for visual telemetry
    setTimeout(() => setActiveGraphStep('retrieve'), 300);
    setTimeout(() => setActiveGraphStep('assess'), 900);
    setTimeout(() => setActiveGraphStep('verify'), 1400);
    setTimeout(() => setActiveGraphStep('synthesize'), 1900);

    try {
      // Build conversational history (last 4 turns)
      const chatHistory = messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(-4)
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          session_id: attachedDoc ? attachedDoc.sessionId : null,
          session_root: attachedDoc ? attachedDoc.sessionRoot : null,
          document_name: attachedDoc ? attachedDoc.documentName : null,
          model_mode: modelMode,
          api_key: apiKey || null,
          chat_history: chatHistory,
          top_k: 5,
          retrieval_k: 10,
        }),
      });

      if (!res.ok) {
        throw new Error('Streaming failed with status: ' + res.status);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep last incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const rawJson = trimmed.substring(6);
            try {
              const payload = JSON.parse(rawJson);

              if (payload.type === 'meta') {
                // Populate metadata & visual pages
                const visualItems = (payload.retrieval_results || []).filter(
                  (r) => r.page_image_url
                );

                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? {
                          ...msg,
                          citations: payload.citations || [],
                          visual_items: visualItems,
                          retrieval_results: payload.retrieval_results || [],
                        }
                      : msg
                  )
                );

                if (payload.retrieval_results?.length) {
                  setInspectorResults(payload.retrieval_results);
                }
              } else if (payload.type === 'token') {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, content: msg.content + payload.token }
                      : msg
                  )
                );
              } else if (payload.type === 'done') {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, isStreaming: false }
                      : msg
                  )
                );
              } else if (payload.type === 'error') {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? {
                          ...msg,
                          content: msg.content + `\n\n*(Error: ${payload.error})*`,
                          isStreaming: false,
                        }
                      : msg
                  )
                );
              }
            } catch (err) {
              console.warn('JSON parse error in SSE stream', err);
            }
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: `⚠️ Mission Control Error: Unable to complete query synthesis (${err.message})`,
                isStreaming: false,
              }
            : msg
        )
      );
    } finally {
      setIsTransmitting(false);
      setActiveGraphStep(null);
    }
  };

  const suggestedQueries = attachedDoc
    ? [
        `Summarize the key payload specifications in ${attachedDoc.documentName}`,
        'What methodologies or system architectures are described in this document?',
        'Explain the primary diagrams and charts shown in this document.',
        'What are the operational margins and performance constraints?',
      ]
    : [
        'What constellation trade-offs apply to LEO precipitation monitoring?',
        'How do multi-frequency radiometers resolve atmospheric moisture and temperature?',
        'Compare CubeSat propulsion architectures for orbital maintenance',
        'What space weather mitigation strategies protect polar orbit payloads?',
      ];

  return (
    <div className="app-container">
      <div className="hud-grid-overlay" />

      {/* TOP MISSION CONTROL HEADER */}
      <header className="hud-header">
        <div className="brand-section">
          <div className="brand-logo-badge">
            <Satellite size={22} />
          </div>
          <div className="brand-info">
            <h1>SPACE RAG</h1>
            <div className="tagline">Mission Control Intelligence Terminal</div>
          </div>
        </div>

        {/* Telemetry Status Indicators */}
        <div className="header-telemetry">
          <div className="telemetry-pill">
            <span className="status-dot" />
            <span style={{ color: '#10b981', fontWeight: 600 }}>SYSTEM NOMINAL</span>
          </div>

          <div className="telemetry-pill" title="Permanent knowledge base index">
            <Database size={13} style={{ color: '#38bdf8' }} />
            <span>CORPUS: {systemStatus?.corpus?.document_count || 2} DOCS</span>
          </div>

          {/* Model Mode Switcher */}
          <div className="model-switch-group">
            <button
              className={`model-switch-btn ${modelMode === 'offline' ? 'active' : ''}`}
              onClick={() => setModelMode('offline')}
              title="Local Qwen 2.5 (1.5B) - Zero data egress"
            >
              <Cpu size={13} />
              Offline Qwen
            </button>
            <button
              className={`model-switch-btn ${modelMode === 'online' ? 'active' : ''}`}
              onClick={() => setModelMode('online')}
              title="xAI Grok-2 Online API"
            >
              <Cloud size={13} />
              Online Grok
            </button>
          </div>

          {modelMode === 'online' && (
            <button
              className="header-action-btn"
              onClick={() => setShowKeyModal(true)}
              title="Configure xAI API Key"
            >
              <Key size={14} style={{ color: apiKey ? '#34d399' : '#f59e0b' }} />
              <span>{apiKey ? 'API Key Active' : 'Set API Key'}</span>
            </button>
          )}

          <button
            className="header-action-btn"
            onClick={handleSyncCorpus}
            disabled={isSyncing}
            title="Synchronize or Rebuild Reference Knowledge Index"
          >
            <RefreshCw size={14} className={isSyncing ? 'spin-anim' : ''} />
            <span>{isSyncing ? 'Syncing...' : 'Sync Index'}</span>
          </button>

          <button
            className={`header-action-btn ${inspectorOpen ? 'active' : ''}`}
            onClick={() => setInspectorOpen(!inspectorOpen)}
            title="Toggle Visual Radar & Evidence Inspector"
          >
            <Radio size={14} />
            <span>Inspector</span>
          </button>
        </div>
      </header>

      {/* WORKSPACE: 3 COLUMNS */}
      <div className="hud-workspace">
        {/* LEFT PANEL: Private Sandbox & Telemetry */}
        <aside className="hud-sidebar">
          {/* SANDBOX SECTION */}
          <div>
            <div className="section-label">
              <Shield size={13} />
              <span>Mission Data Sandbox</span>
            </div>

            {attachedDoc ? (
              <div className="active-doc-card">
                <div className="active-doc-header">
                  <span className="active-doc-tag">
                    <span className="status-dot" style={{ width: 6, height: 6 }} />
                    ACTIVE PRIVATE ATTACHMENT
                  </span>
                </div>
                <div className="active-doc-name">{attachedDoc.documentName}</div>
                <div className="active-doc-meta">
                  <span>📄 {attachedDoc.totalPages} Pages</span>
                  <span>🔒 Isolated Sandbox</span>
                </div>
                <button className="detach-btn" onClick={handleDetachDoc}>
                  <Trash2 size={13} />
                  Detach & Purge Document
                </button>
              </div>
            ) : (
              <div
                className={`upload-sandbox ${isDragging ? 'drag-over' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  if (e.dataTransfer.files?.[0]) {
                    handleFileUpload(e.dataTransfer.files[0]);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept=".pdf"
                  onChange={(e) => {
                    if (e.target.files?.[0]) {
                      handleFileUpload(e.target.files[0]);
                    }
                  }}
                />
                <Upload size={24} className="upload-icon" />
                <div className="upload-title">
                  {isUploading ? 'Ingesting PDF...' : 'Attach Flight Spec'}
                </div>
                <div className="upload-subtitle">
                  Drag & drop PDF for private analysis
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* CENTER PANEL: Interactive Chat & Command Viewport */}
        <main className="hud-terminal-view">
          <div className="chat-messages-scroll">
            {messages.length === 0 ? (
              <div className="empty-terminal-state">
                <div className="empty-radar-sphere">
                  <Satellite size={40} />
                </div>
                <h2 className="empty-title">Orbit Initialized</h2>
                <p className="empty-subtitle">
                  Ask queries regarding orbital mechanics, telemetry tables, radar radiometer architectures, or upload flight manuals for private source-grounded analysis.
                </p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                  {suggestedQueries.slice(0, 2).map((q, i) => (
                    <button
                      key={i}
                      className="suggested-prompt-btn"
                      style={{ maxWidth: '320px' }}
                      onClick={() => handleTransmit(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => {
                if (msg.role === 'system') {
                  return (
                    <div
                      key={msg.id}
                      style={{
                        padding: '8px 16px',
                        background: 'rgba(15, 23, 42, 0.7)',
                        border: '1px solid rgba(56, 189, 248, 0.2)',
                        borderRadius: '8px',
                        fontSize: '0.8rem',
                        fontFamily: 'var(--font-mono)',
                        color: '#7dd3fc',
                        alignSelf: 'center',
                      }}
                    >
                      {msg.content}
                    </div>
                  );
                }

                const isUser = msg.role === 'user';

                return (
                  <div
                    key={msg.id}
                    className={`chat-row ${isUser ? 'user-row' : 'assistant-row'}`}
                  >
                    <div
                      className={`avatar-badge ${
                        isUser ? 'user-avatar' : 'assistant-avatar'
                      }`}
                    >
                      {isUser ? '🧑‍🚀' : <Satellite size={18} />}
                    </div>

                    <div className="message-bubble">
                      {/* State Machine Step Tracker during generation */}
                      {!isUser && msg.isStreaming && activeGraphStep && (
                        <div className="graph-pipeline-tracker">
                          <span
                            className={`pipeline-step ${
                              ['retrieve', 'assess', 'verify', 'synthesize'].includes(
                                activeGraphStep
                              )
                                ? 'completed'
                                : 'active'
                            }`}
                          >
                            Understand
                          </span>
                          <span className="pipeline-arrow">➔</span>
                          <span
                            className={`pipeline-step ${
                              ['assess', 'verify', 'synthesize'].includes(
                                activeGraphStep
                              )
                                ? 'completed'
                                : activeGraphStep === 'retrieve'
                                ? 'active'
                                : ''
                            }`}
                          >
                            Retrieve
                          </span>
                          <span className="pipeline-arrow">➔</span>
                          <span
                            className={`pipeline-step ${
                              ['verify', 'synthesize'].includes(activeGraphStep)
                                ? 'completed'
                                : activeGraphStep === 'assess'
                                ? 'active'
                                : ''
                            }`}
                          >
                            Assess
                          </span>
                          <span className="pipeline-arrow">➔</span>
                          <span
                            className={`pipeline-step ${
                              activeGraphStep === 'synthesize'
                                ? 'completed'
                                : activeGraphStep === 'verify'
                                ? 'active'
                                : ''
                            }`}
                          >
                            Verify
                          </span>
                          <span className="pipeline-arrow">➔</span>
                          <span
                            className={`pipeline-step ${
                              activeGraphStep === 'synthesize' ? 'active' : ''
                            }`}
                          >
                            Synthesize
                          </span>
                        </div>
                      )}

                      {/* Content rendering */}
                      <div style={{ whiteSpace: 'pre-wrap' }}>
                        {msg.content}
                        {msg.isStreaming && (
                          <span className="typing-indicator">
                            <span className="typing-dot" />
                            <span className="typing-dot" />
                            <span className="typing-dot" />
                          </span>
                        )}
                      </div>

                      {/* Citations Badges */}
                      {msg.citations?.length > 0 && (
                        <div className="citations-container">
                          <span className="citations-label">📎 Private Sources:</span>
                          {msg.citations.map((c, i) => (
                            <span
                              key={i}
                              className="citation-chip"
                              onClick={() => {
                                setInspectorResults(msg.retrieval_results || []);
                                setInspectorOpen(true);
                              }}
                              title="Inspect evidence details in right panel"
                            >
                              {c}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Referenced Document Visual Pages */}
                      {msg.visual_items?.length > 0 && (
                        <div className="visual-evidence-drawer">
                          <div className="visual-drawer-header">
                            <span>🖼️ REFERENCED MISSION VISUAL PAGES</span>
                            <span style={{ fontSize: '0.68rem', color: '#94a3b8' }}>
                              Click to expand 150 DPI view
                            </span>
                          </div>
                          <div className="visual-cards-row">
                            {msg.visual_items.slice(0, 3).map((item, idx) => (
                              <div
                                key={idx}
                                className="visual-thumb-card"
                                onClick={() =>
                                  setLightboxImage({
                                    url: item.page_image_url,
                                    title: `${item.document || 'Document'} — Page ${
                                      item.page_number || '?'
                                    }`,
                                  })
                                }
                              >
                                <img
                                  src={item.page_image_url}
                                  alt="Mission chart"
                                  loading="lazy"
                                />
                                <div className="visual-thumb-caption">
                                  Page {item.page_number}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* COMMAND PROMPT BAR */}
          <div className="hud-prompt-bar">
            <div className="prompt-input-container">
              <span className="prompt-prefix">&gt;</span>
              <input
                type="text"
                className="prompt-text-input"
                placeholder={
                  attachedDoc
                    ? `Query ${attachedDoc.documentName} or aerospace corpus...`
                    : 'Query satellite systems, remote sensing, or orbit mechanics...'
                }
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleTransmit();
                  }
                }}
                disabled={isTransmitting}
              />
            </div>
            <button
              className="transmit-btn"
              onClick={() => handleTransmit()}
              disabled={isTransmitting || !queryInput.trim()}
              title="Transmit Query (Enter)"
            >
              <Send size={18} />
            </button>
          </div>
        </main>

        {/* RIGHT PANEL: Radar & Evidence Inspector Drawer */}
        {inspectorOpen && (
          <aside className="hud-inspector-drawer">
            <div className="inspector-header">
              <div className="inspector-title">
                <Radio size={16} style={{ color: '#00e5ff' }} />
                <span>Evidence Inspector</span>
              </div>
              <button
                className="close-inspector-btn"
                onClick={() => setInspectorOpen(false)}
              >
                <X size={16} />
              </button>
            </div>

            {inspectorResults.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '30px 10px', color: '#64748b', fontSize: '0.85rem' }}>
                <Layers size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                <p>Run a query to inspect candidate evidence, fusion scores, and cross-modal verification.</p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                  TOP RETRIEVED CANDIDATES ({inspectorResults.length})
                </div>

                {inspectorResults.map((candidate, idx) => (
                  <div key={idx} className="evidence-candidate-card">
                    <div className="candidate-header">
                      <span className="candidate-rank">#{idx + 1} RANK</span>
                      <span className="candidate-score-badge">
                        Score: {candidate.fusion_score}
                      </span>
                    </div>

                    <div className="candidate-doc">
                      {candidate.source_type === 'private'
                        ? `📎 [Private] ${candidate.document} (p. ${candidate.page_number})`
                        : `📚 Permanent Aerospace Knowledge (p. ${candidate.page_number})`}
                    </div>

                    {candidate.text && (
                      <div className="candidate-snippet">{candidate.text}</div>
                    )}

                    {candidate.page_image_url && (
                      <div
                        style={{
                          borderRadius: 6,
                          overflow: 'hidden',
                          border: '1px solid rgba(56, 189, 248, 0.3)',
                          cursor: 'pointer',
                          marginTop: 4,
                        }}
                        onClick={() =>
                          setLightboxImage({
                            url: candidate.page_image_url,
                            title: `Page ${candidate.page_number} Visual Inspection`,
                          })
                        }
                      >
                        <img
                          src={candidate.page_image_url}
                          alt="Evidence page"
                          style={{ width: '100%', height: '110px', objectFit: 'cover' }}
                        />
                      </div>
                    )}

                    {candidate.verification && (
                      <div
                        className={`verification-badge ${
                          candidate.verification.status === 'VERIFIED'
                            ? 'verified'
                            : 'contradicted'
                        }`}
                      >
                        <Shield size={11} />
                        <span>
                          {candidate.verification.status} (
                          {Math.round(
                            (candidate.verification.confidence || 0) * 100
                          )}
                          %)
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </aside>
        )}
      </div>

      {/* LIGHTBOX MODAL FOR 150 DPI VISUAL INSPECTION */}
      {lightboxImage && (
        <div className="lightbox-modal" onClick={() => setLightboxImage(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <div className="lightbox-header">
              <span style={{ fontSize: '0.88rem', fontWeight: 600, color: '#f8fafc' }}>
                {lightboxImage.title}
              </span>
              <button
                className="close-inspector-btn"
                onClick={() => setLightboxImage(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="lightbox-body">
              <img src={lightboxImage.url} alt="Expanded Inspection" />
            </div>
          </div>
        </div>
      )}

      {/* API KEY CONFIGURATION MODAL */}
      {showKeyModal && (
        <div className="lightbox-modal" onClick={() => setShowKeyModal(false)}>
          <div
            className="lightbox-content"
            style={{ width: '420px', padding: 20 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="lightbox-header" style={{ padding: '0 0 12px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Key size={18} style={{ color: '#00e5ff' }} />
                <span style={{ fontWeight: 600, color: '#fff' }}>
                  xAI Grok API Configuration
                </span>
              </div>
              <button
                className="close-inspector-btn"
                onClick={() => setShowKeyModal(false)}
              >
                <X size={18} />
              </button>
            </div>
            <div style={{ padding: '14px 0', fontSize: '0.85rem', color: '#94a3b8' }}>
              Enter your xAI API key for online Grok-2 inference. Stored strictly in memory for this session:
              <input
                type="password"
                className="prompt-text-input"
                placeholder="xai-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                style={{
                  width: '100%',
                  marginTop: 10,
                  padding: '8px 12px',
                  background: 'rgba(8, 14, 28, 0.9)',
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  borderRadius: 6,
                  color: '#fff',
                }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                className="header-action-btn"
                onClick={() => setShowKeyModal(false)}
              >
                Save & Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
