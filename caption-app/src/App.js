import React, { useRef, useState } from 'react';
import axios from 'axios';

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
    if (isProcessingRef.current) {
      return;
    }

    setLastProcessedImage(imageBase64);
    setIsProcessing(true);
    isProcessingRef.current = true;
    setError('');

    // Cancel previous request if still running
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
          timeout: 10000
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
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        return;
      }
      setError(err.response?.data?.error || err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
      isProcessingRef.current = false;
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
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

  return (
    <div style={{ 
      textAlign: 'center', 
      padding: '20px',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <h1>Image Captioning</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="captionMode" style={{ marginRight: '8px', fontWeight: 600 }}>
          Caption Type:
        </label>
        <select
          id="captionMode"
          value={captionMode}
          onChange={(e) => setCaptionMode(e.target.value)}
          style={{
            padding: '10px 12px',
            fontSize: '14px',
            borderRadius: '8px',
            border: '1px solid #d1d5db',
            minWidth: '360px'
          }}
        >
          {captionModeOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div style={{ 
        display: 'inline-block', 
        position: 'relative',
        borderRadius: '8px',
        overflow: 'hidden',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <div style={{
          width: '640px',
          height: '480px',
          backgroundColor: '#f3f4f6',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          border: '2px dashed #d1d5db'
        }}>
          {uploadedImage ? (
            <img 
              src={uploadedImage} 
              alt="Uploaded" 
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain'
              }}
            />
          ) : (
            <div style={{ textAlign: 'center', color: '#6b7280' }}>
              <p style={{ fontSize: '48px', margin: '0 0 10px 0' }}>📷</p>
              <p style={{ fontSize: '16px', margin: '0' }}>No image selected</p>
            </div>
          )}
        </div>
        {isProcessing && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '4px',
            fontSize: '12px'
          }}>
            Processing...
          </div>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            backgroundColor: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#2563eb'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#3b82f6'}
        >
          {uploadedImage ? '🔄 Change Photo' : '📁 Choose Photo'}
        </button>
        {uploadedImage && (
          <button
            onClick={() => {
              setUploadedImage(null);
              setCaption('');
              setError('');
              setLastProcessedImage(null);
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
              }
            }}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '600',
              marginLeft: '10px',
              transition: 'background-color 0.2s'
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = '#dc2626'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#ef4444'}
          >
            ✕ Clear
          </button>
        )}
      </div>

      <div style={{ marginTop: '20px' }}>
        <h2 style={{ 
          color: error ? '#dc2626' : '#1f2937',
          minHeight: '40px'
        }}>
          {error ? `⚠️ ${error}` : caption ? `📷 ${caption}` : '📁 Select a photo to generate caption'}
        </h2>
      </div>

      {lastProcessedImage && (
        <div style={{ marginTop: '16px' }}>
          <button
            onClick={() => processImage(lastProcessedImage)}
            disabled={isProcessing}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: isProcessing ? '#9ca3af' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              transition: 'background-color 0.2s'
            }}
            onMouseOver={(e) => !isProcessing && (e.target.style.backgroundColor = '#059669')}
            onMouseOut={(e) => !isProcessing && (e.target.style.backgroundColor = '#10b981')}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
