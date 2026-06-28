import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

/* ═══════════════════════════════════════════════════════════════
   AI VOICE ONBOARDING — MAIN APPLICATION
   ═══════════════════════════════════════════════════════════════ */

// ── Constants ────────────────────────────────────────────────
const PROVIDERS = [
  { value: 'openai', label: 'OpenAI', icon: '🤖' },
  { value: 'gemini', label: 'Gemini', icon: '✨' },
  { value: 'anthropic', label: 'Anthropic', icon: '🧠' },
  { value: 'ollama', label: 'Ollama', icon: '🦙' },
  { value: 'openrouter', label: 'OpenRouter', icon: '🔀' },
  { value: 'minimax', label: 'MiniMax (NVIDIA)', icon: '🟢' },
];

const PIPELINE_STEPS = [
  { id: 'preprocessing', label: 'Audio Preprocessing', icon: '🎵' },
  { id: 'stt', label: 'Speech-to-Text', icon: '📝' },
  { id: 'normalization', label: 'Text Normalization', icon: '✏️' },
  { id: 'analysis', label: 'AI Analysis', icon: '🔬' },
  { id: 'aggregation', label: 'Data Aggregation', icon: '📊' },
  { id: 'llm', label: 'LLM Profile Generation', icon: '🧠' },
  { id: 'profile', label: 'Profile Assembly', icon: '👤' },
];

const ACCEPTED_AUDIO_TYPES = [
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/x-wav',
  'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/mp4', 'audio/aac',
  'audio/webm', 'audio/x-m4a',
];

const rawApiUrl = import.meta.env.VITE_API_URL || '';
const backendBaseUrl = rawApiUrl && !rawApiUrl.startsWith('http') 
  ? `https://${rawApiUrl}` 
  : rawApiUrl;
const API_ENDPOINT = backendBaseUrl ? `${backendBaseUrl}/api/process` : '/api/process';

// ── Helpers ──────────────────────────────────────────────────
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileExtension(filename) {
  return filename.split('.').pop().toUpperCase();
}

// ── Sparkle Positions (for hero) ─────────────────────────────
const sparklePositions = [
  { top: '15%', left: '10%', delay: '0s', duration: '4s' },
  { top: '25%', right: '15%', delay: '1s', duration: '3.5s' },
  { top: '60%', left: '20%', delay: '0.5s', duration: '5s' },
  { top: '40%', right: '25%', delay: '1.5s', duration: '4.5s' },
  { top: '75%', left: '70%', delay: '2s', duration: '3s' },
  { top: '10%', left: '50%', delay: '0.8s', duration: '4.2s' },
  { top: '85%', right: '10%', delay: '2.5s', duration: '3.8s' },
];


// ═══════════════════════════════════════════════════════════════
//  SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════

// ── Hero / Header ────────────────────────────────────────────
function HeroSection() {
  return (
    <header className="hero" id="hero-section">
      <div className="hero__sparkles">
        {sparklePositions.map((pos, i) => (
          <span
            key={i}
            className="hero__sparkle"
            style={{
              ...pos,
              animationDelay: pos.delay,
              animationDuration: pos.duration,
            }}
          />
        ))}
      </div>

      <div className="hero__badge">
        <span className="hero__badge-dot" />
        AI-Powered Voice Analysis
      </div>

      <h1 className="hero__title">
        Discover Your{' '}
        <span className="hero__title-highlight">Dating Personality</span>
      </h1>
    </header>
  );
}


// ── Configuration Panel ──────────────────────────────────────
function ConfigPanel({ provider, setProvider, apiKey, setApiKey }) {
  return (
    <section className="config-panel" id="config-section">
      <div className="section-label">
        <span className="section-label__icon">⚙️</span>
        Configuration
      </div>

      <div className="config-panel__grid">
        <div className="config-panel__field">
          <label className="config-panel__label" htmlFor="provider-select">
            LLM Provider
          </label>
          <select
            id="provider-select"
            className="config-panel__select"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.icon} {p.label}
              </option>
            ))}
          </select>
        </div>

        <div className="config-panel__field">
          <label className="config-panel__label" htmlFor="api-key-input">
            API Key
          </label>
          <input
            id="api-key-input"
            className="config-panel__input"
            type="password"
            placeholder="Enter your API key…"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </div>
      </div>
    </section>
  );
}


// ── Waveform Visualizer ──────────────────────────────────────
function WaveformPreview() {
  const barCount = 40;
  return (
    <div className="upload-zone__waveform" aria-hidden="true">
      {Array.from({ length: barCount }).map((_, i) => {
        const height = 8 + Math.sin(i * 0.5) * 16 + Math.random() * 12;
        return (
          <span
            key={i}
            className="upload-zone__waveform-bar"
            style={{
              height: `${height}px`,
              animationDelay: `${i * 0.05}s`,
              opacity: 0.5 + Math.sin(i * 0.3) * 0.3,
            }}
          />
        );
      })}
    </div>
  );
}


// ── Audio Upload Zone ────────────────────────────────────────
function UploadZone({ file, setFile }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback((files) => {
    if (files && files.length > 0) {
      const f = files[0];
      if (ACCEPTED_AUDIO_TYPES.includes(f.type) || f.name.match(/\.(mp3|wav|ogg|flac|m4a|aac|webm)$/i)) {
        setFile(f);
      } else {
        alert('Please upload a valid audio file (MP3, WAV, OGG, FLAC, M4A, AAC, or WebM).');
      }
    }
  }, [setFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const dzClasses = [
    'upload-zone__dropzone',
    isDragOver && 'upload-zone__dropzone--dragover',
    file && 'upload-zone__dropzone--has-file',
  ].filter(Boolean).join(' ');

  return (
    <section className="upload-zone" id="upload-section">
      <div className="section-label">
        <span className="section-label__icon">🎤</span>
        Voice Upload
      </div>

      <div
        className={dzClasses}
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Upload audio file"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <div className="upload-zone__icon-wrapper">
          <div className="upload-zone__icon-bg" />
          <span className="upload-zone__icon">
            {file ? '✅' : isDragOver ? '📥' : '🎙️'}
          </span>
        </div>

        <p className="upload-zone__title">
          {file
            ? 'File ready for analysis'
            : isDragOver
              ? 'Drop your audio file here!'
              : 'Drag & drop your voice clip here'}
        </p>
        <p className="upload-zone__subtitle">
          {file
            ? 'Click to replace or drag a new file'
            : <>or <strong>click to browse</strong> — MP3, WAV, OGG, FLAC, M4A, WebM</>}
        </p>

        <input
          ref={inputRef}
          type="file"
          className="upload-zone__input"
          accept="audio/*"
          onChange={(e) => handleFiles(e.target.files)}
          id="audio-file-input"
        />

        {file && (
          <div className="upload-zone__file-info" onClick={(e) => e.stopPropagation()}>
            <div className="upload-zone__file-icon">🎵</div>
            <div className="upload-zone__file-details">
              <div className="upload-zone__file-name">{file.name}</div>
              <div className="upload-zone__file-size">
                {formatFileSize(file.size)} • {getFileExtension(file.name)}
              </div>
            </div>
            <button
              className="upload-zone__file-remove"
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
                if (inputRef.current) inputRef.current.value = '';
              }}
              aria-label="Remove file"
              title="Remove file"
            >
              ✕
            </button>
          </div>
        )}

        {file && <WaveformPreview />}
      </div>
    </section>
  );
}


// ── Pipeline Status ──────────────────────────────────────────
function PipelineStatus({ currentStep }) {
  return (
    <section className="pipeline" id="pipeline-section">
      <div className="section-label">
        <span className="section-label__icon">⚡</span>
        Processing Pipeline
      </div>

      <div className="pipeline__steps">
        {PIPELINE_STEPS.map((step, index) => {
          let status = 'pending';
          if (index < currentStep) status = 'completed';
          else if (index === currentStep) status = 'active';

          return (
            <div key={step.id}>
              <div className={`pipeline__step pipeline__step--${status}`}>
                <div className="pipeline__step-indicator">
                  {status === 'completed' ? '✓' : status === 'active' ? step.icon : (index + 1)}
                </div>
                <span className="pipeline__step-label">{step.label}</span>
                {status !== 'pending' && (
                  <span className="pipeline__step-status">
                    {status === 'active' ? 'Processing…' : 'Done'}
                  </span>
                )}
              </div>
              {index < PIPELINE_STEPS.length - 1 && (
                <div className={`pipeline__connector pipeline__connector--${index < currentStep ? 'completed' : 'pending'}`} />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}


// ── Result Card ──────────────────────────────────────────────
function ResultCard({ variant, icon, title, children, className = '' }) {
  return (
    <article className={`result-card result-card--${variant} ${className}`}>
      <div className="result-card__header">
        <span className="result-card__icon">{icon}</span>
        <h3 className="result-card__title">{title}</h3>
      </div>
      <div className="result-card__content">
        {children}
      </div>
    </article>
  );
}


// ── Audio Metrics Sub-Component ──────────────────────────────
function AudioMetrics({ metrics }) {
  if (!metrics || typeof metrics !== 'object') return null;
  const entries = Object.entries(metrics);
  if (entries.length === 0) return null;

  return (
    <div className="result-card__metrics">
      {entries.map(([key, value]) => {
        const numVal = typeof value === 'number' ? value : parseFloat(value);
        const isPercentLike = !isNaN(numVal) && numVal >= 0 && numVal <= 1;
        const displayVal = isPercentLike
          ? `${(numVal * 100).toFixed(0)}%`
          : typeof value === 'number'
            ? numVal.toFixed(1)
            : String(value);

        return (
          <div key={key} className="result-card__metric">
            <span className="result-card__metric-label">
              {key.replace(/_/g, ' ')}
            </span>
            <span className="result-card__metric-value">{displayVal}</span>
            {isPercentLike && (
              <div className="result-card__metric-bar">
                <div
                  className="result-card__metric-bar-fill"
                  style={{ width: `${numVal * 100}%` }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}


// ── Render helpers for result card content ────────────────────
function renderTextOrList(data) {
  if (!data) return <p style={{ color: 'var(--color-text-muted)' }}>No data available</p>;
  if (typeof data === 'string') return <p>{data}</p>;
  if (Array.isArray(data)) {
    return (
      <ul className="result-card__list">
        {data.map((item, i) => (
          <li key={i} className="result-card__list-item">
            <span className="result-card__list-bullet">•</span>
            <span>{typeof item === 'object' ? JSON.stringify(item) : item}</span>
          </li>
        ))}
      </ul>
    );
  }
  if (typeof data === 'object') {
    return (
      <ul className="result-card__list">
        {Object.entries(data).map(([k, v]) => (
          <li key={k} className="result-card__list-item">
            <span className="result-card__list-bullet">▸</span>
            <span><strong>{k.replace(/_/g, ' ')}:</strong> {typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
          </li>
        ))}
      </ul>
    );
  }
  return <p>{String(data)}</p>;
}

function renderTags(data, variant) {
  if (!data) return null;
  const items = Array.isArray(data) ? data : [data];
  return (
    <div className="result-card__tags">
      {items.map((item, i) => (
        <span key={i} className={`result-card__tag ${variant ? `result-card__tag--${variant}` : ''}`}>
          {typeof item === 'object' ? JSON.stringify(item) : item}
        </span>
      ))}
    </div>
  );
}


// ── Results Dashboard ────────────────────────────────────────
function ResultsDashboard({ results }) {
  const downloadResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voice-profile-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Attempt to extract fields with flexible key matching
  const get = (obj, ...keys) => {
    if (!obj) return undefined;
    for (const key of keys) {
      // Exact match
      if (obj[key] !== undefined) return obj[key];
      // Case-insensitive match
      const lower = key.toLowerCase();
      for (const k of Object.keys(obj)) {
        if (k.toLowerCase() === lower) return obj[k];
        if (k.toLowerCase().replace(/[_\s]/g, '') === lower.replace(/[_\s]/g, '')) return obj[k];
      }
    }
    return undefined;
  };

  const dp = results.dating_profile || {};
  const personality = get(dp, 'personality_summary', 'personality', 'personalitySummary') || get(results, 'personality_summary');
  const bio = get(dp, 'dating_bio', 'bio', 'datingBio') || get(results, 'dating_bio');
  const compatibility = get(dp, 'compatibility_features', 'compatibility', 'compatibilityFeatures') || get(results, 'compatibility_features');
  const iceBreakers = get(dp, 'ice_breakers', 'iceBreakers', 'icebreakers', 'conversation_starters') || get(results, 'ice_breakers');
  const greenFlags = get(dp, 'green_flags', 'greenFlags', 'greenlights') || get(results, 'green_flags');
  const redFlags = get(dp, 'red_flags', 'redFlags', 'redlights') || get(results, 'red_flags');
  const conversationStyle = get(dp, 'conversation_style', 'conversationStyle', 'communication_style') || get(results, 'conversation_style');
  const matchFeatures = get(dp, 'match_recommendations', 'match_recommendation', 'matchRecommendation', 'match_features', 'matchFeatures') || get(results, 'match_recommendations');
  
  const audioIntel = get(results, 'audio_features', 'audio_intelligence', 'audioIntelligence', 'audio_metrics', 'audioMetrics');
  const transcriptIntel = get(results, 'conversation_summary', 'transcript_intelligence', 'transcriptIntelligence', 'transcript_analysis', 'transcriptAnalysis');

  return (
    <section className="results-dashboard" id="results-section">
      <div className="results-dashboard__header">
        <h2 className="results-dashboard__title">
          <span className="results-dashboard__title-icon">🌻</span>
          Your Voice Profile
        </h2>
        <button
          className="download-btn"
          onClick={downloadResults}
          id="download-results-btn"
        >
          <span className="download-btn__icon">📥</span>
          Download JSON
        </button>
      </div>

      <div className="results-grid">
        {/* Personality Summary */}
        <ResultCard variant="personality" icon="🌟" title="Personality Summary">
          {renderTextOrList(personality)}
        </ResultCard>

        {/* Dating Bio */}
        <ResultCard variant="bio" icon="💛" title="Dating Bio">
          {renderTextOrList(bio)}
        </ResultCard>

        {/* Two-column: Compatibility + Conversation Style */}
        <div className="results-grid--two-col">
          <ResultCard variant="compatibility" icon="🧩" title="Compatibility Features">
            {renderTags(compatibility)}
            {!Array.isArray(compatibility) && renderTextOrList(compatibility)}
          </ResultCard>

          <ResultCard variant="conversation" icon="💬" title="Conversation Style">
            {renderTextOrList(conversationStyle)}
          </ResultCard>
        </div>

        {/* Ice Breakers */}
        <ResultCard variant="icebreakers" icon="🧊" title="Ice Breakers">
          {renderTextOrList(iceBreakers)}
        </ResultCard>

        {/* Two-column: Green Flags + Red Flags */}
        <div className="results-grid--two-col">
          <ResultCard variant="greenflags" icon="🟢" title="Green Flags">
            {renderTags(greenFlags, 'green')}
            {!Array.isArray(greenFlags) && renderTextOrList(greenFlags)}
          </ResultCard>

          <ResultCard variant="redflags" icon="🔴" title="Red Flags">
            {renderTags(redFlags, 'red')}
            {!Array.isArray(redFlags) && renderTextOrList(redFlags)}
          </ResultCard>
        </div>

        {/* Match Recommendation */}
        <ResultCard variant="match" icon="💘" title="Match Recommendation">
          {renderTextOrList(matchFeatures)}
        </ResultCard>

        {/* Audio Intelligence */}
        <ResultCard variant="audio" icon="🎧" title="Audio Intelligence">
          {audioIntel && typeof audioIntel === 'object' && !Array.isArray(audioIntel)
            ? <AudioMetrics metrics={audioIntel} />
            : renderTextOrList(audioIntel)}
        </ResultCard>

        {/* Transcript Intelligence */}
        <ResultCard variant="transcript" icon="📜" title="Transcript Intelligence">
          {renderTextOrList(transcriptIntel)}
        </ResultCard>
      </div>
    </section>
  );
}


// ═══════════════════════════════════════════════════════════════
//  MAIN APP COMPONENT
// ═══════════════════════════════════════════════════════════════
function App() {
  // State
  const [provider, setProvider] = useState('openai');
  const [apiKey, setApiKey] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Pipeline simulation timer ref
  const pipelineTimerRef = useRef(null);

  // Cleanup pipeline timer on unmount
  useEffect(() => {
    return () => {
      if (pipelineTimerRef.current) clearInterval(pipelineTimerRef.current);
    };
  }, []);

  // ── Process Handler ──────────────────────────────────────────
  const handleProcess = useCallback(async () => {
    // Validation
    if (!audioFile) {
      setError('Please upload an audio file first.');
      return;
    }
    if (!apiKey.trim()) {
      setError('Please enter your API key.');
      return;
    }

    setError(null);
    setResults(null);
    setIsProcessing(true);
    setPipelineStep(0);

    // Simulate pipeline progression while waiting for API
    let step = 0;
    pipelineTimerRef.current = setInterval(() => {
      step += 1;
      if (step < PIPELINE_STEPS.length) {
        setPipelineStep(step);
      }
    }, 2500);

    try {
      const formData = new FormData();
      formData.append('audio', audioFile);
      formData.append('api_key', apiKey.trim());
      formData.append('provider', provider);

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        body: formData,
      });

      // Clear pipeline timer and show all steps complete
      clearInterval(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
      setPipelineStep(PIPELINE_STEPS.length);

      if (!response.ok) {
        let errorMessage = `Server error (${response.status})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorData.error || errorMessage;
        } catch {
          // If response isn't JSON, use the status text
          errorMessage = `${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setResults(data);

      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 300);

    } catch (err) {
      clearInterval(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
      setPipelineStep(-1);

      if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
        const displayUrl = API_ENDPOINT.startsWith('http') ? API_ENDPOINT : window.location.origin + API_ENDPOINT;
        setError(`Could not connect to the server. Make sure the backend is running at ${displayUrl}`);
      } else {
        setError(err.message || 'An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  }, [audioFile, apiKey, provider]);

  // ── Can process? ──────────────────────────────────────────
  const canProcess = audioFile && apiKey.trim() && !isProcessing;

  // ── Render ────────────────────────────────────────────────
  return (
    <div className="app" id="app-root">
      {/* Decorative background orbs */}
      <div className="app__background-orb app__background-orb--green" aria-hidden="true" />
      <div className="app__background-orb app__background-orb--yellow" aria-hidden="true" />
      <div className="app__background-orb app__background-orb--accent" aria-hidden="true" />

      {/* Hero */}
      <HeroSection />

      {/* Main Content */}
      <main className="main-content" id="main-content">
        {/* Configuration */}
        <ConfigPanel
          provider={provider}
          setProvider={setProvider}
          apiKey={apiKey}
          setApiKey={setApiKey}
        />

        {/* Upload */}
        <UploadZone file={audioFile} setFile={setAudioFile} />

        {/* Error Alert */}
        {error && (
          <div className="error-alert" id="error-alert" role="alert">
            <span className="error-alert__icon">⚠️</span>
            <p className="error-alert__text">{error}</p>
            <button
              className="error-alert__dismiss"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        )}

        {/* Process Button */}
        <div className="process-btn-container">
          <button
            className="process-btn"
            onClick={handleProcess}
            disabled={!canProcess}
            id="process-btn"
          >
            {isProcessing ? (
              <>
                <span className="process-btn__spinner" />
                Analyzing Voice…
              </>
            ) : (
              <>
                <span className="process-btn__icon">🌻</span>
                Analyze My Voice
              </>
            )}
          </button>
        </div>

        {/* Pipeline Status */}
        {isProcessing && pipelineStep >= 0 && (
          <PipelineStatus currentStep={pipelineStep} />
        )}

        {/* Results Dashboard */}
        {results && <ResultsDashboard results={results} />}
      </main>

    </div>
  );
}

export default App;
