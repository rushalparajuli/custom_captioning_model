import React, { useRef, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const fileInputRef = useRef(null);
  const [caption, setCaption] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [captionMode, setCaptionMode] = useState('consistent');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [lastProcessedImage, setLastProcessedImage] = useState(null);
  const abortControllerRef = useRef(null);
  const isProcessingRef = useRef(false);

  const captionModeOptions = [
    { value: 'consistent', label: 'Consistent caption (beam search, width 5)' },
    { value: 'safe_diverse', label: 'Safe diverse caption' },
    { value: 'balanced_diverse', label: 'Balanced diverse captions' },
    { value: 'creative_diverse', label: 'Creative diverse caption' },
  ];

  const processImage = async (imageBase64) => {
    if (isProcessingRef.current) return;

    setLastProcessedImage(imageBase64);
    setIsProcessing(true);
    isProcessingRef.current = true;
    setError('');

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await axios.post(
        'http://localhost:8080/process',
        {
          image_base64: imageBase64,
          caption_mode: captionMode,
        },
        {
          signal: abortControllerRef.current.signal,
          timeout: 10000,
        }
      );

      const newCaption = response.data.caption;

      if (newCaption) {
        setCaption(newCaption);
        window.speechSynthesis.cancel();

        setTimeout(() => {
          const utterance = new SpeechSynthesisUtterance(newCaption);
          utterance.rate = 1.0;
          utterance.pitch = 1.0;
          utterance.volume = 1.0;
          window.speechSynthesis.speak(utterance);
        }, 100);
      }
    } catch (err) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') return;
      setError(err.response?.data?.error || err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
      isProcessingRef.current = false;
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result;
      setUploadedImage(base64String);
      processImage(base64String);
    };
    reader.onerror = () => {
      setError('Failed to read file');
    };
    reader.readAsDataURL(file);
  };

  const handleClear = () => {
    setUploadedImage(null);
    setCaption('');
    setError('');
    setLastProcessedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const captionText = error
    ? `⚠️ ${error}`
    : caption
      ? `📷 ${caption}`
      : '📁 Select a photo to generate caption';

  return (
    <div className="app">
      <h1 className="app__title">Image Captioning</h1>

      <div className="app__card">

        {/* ── Caption mode selector ── */}
        <div className="caption-mode">
          <label className="caption-mode__label" htmlFor="captionMode">
            Caption Type
          </label>
          <select
            id="captionMode"
            className="caption-mode__select"
            value={captionMode}
            onChange={(e) => setCaptionMode(e.target.value)}
          >
            {captionModeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* ── Image preview ── */}
        <div className="image-box">
          {uploadedImage ? (
            <img className="image-box__img" src={uploadedImage} alt="Uploaded" />
          ) : (
            <div className="image-box__placeholder">
              <span className="image-box__placeholder-icon">📷</span>
              <span className="image-box__placeholder-text">No image selected</span>
            </div>
          )}
          {isProcessing && (
            <div className="image-box__processing-badge">Processing…</div>
          )}
        </div>

        {/* ── Action buttons ── */}
        <div className="button-row">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            className="btn btn--primary"
            onClick={() => fileInputRef.current?.click()}
          >
            {uploadedImage ? '🔄 Change Photo' : '📁 Choose Photo'}
          </button>

          {uploadedImage && (
            <button className="btn btn--danger" onClick={handleClear}>
              ✕ Clear
            </button>
          )}
        </div>

        {/* ── Caption output ── */}
        <div className="caption-output">
          <p className={`caption-output__text${error ? ' caption-output__text--error' : ''}`}>
            {captionText}
          </p>
        </div>

        {/* ── Retry ── */}
        {lastProcessedImage && (
          <div className="retry-row">
            <button
              className="btn btn--success"
              onClick={() => processImage(lastProcessedImage)}
              disabled={isProcessing}
            >
              🔁 Try again
            </button>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;