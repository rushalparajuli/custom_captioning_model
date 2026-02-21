import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

function App() {
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const [caption, setCaption] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [mode, setMode] = useState('webcam'); // 'webcam' or 'upload'
  const [uploadedImage, setUploadedImage] = useState(null);
  const [lastProcessedImage, setLastProcessedImage] = useState(null); // same image for "Try again"
  const abortControllerRef = useRef(null);
  const isProcessingRef = useRef(false); // Use ref to avoid dependency issues

  // Function to process image and generate caption
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
      console.log('Sending request to Go server...');
      const response = await axios.post(
        'http://localhost:8080/process',
        { image_base64: imageBase64 },
        { 
          signal: abortControllerRef.current.signal,
          timeout: 10000 // Increased to 10 seconds
        }
      );
      console.log('Received response:', response.data);

      const newCaption = response.data.caption;
      console.log('Caption received:', newCaption);
      
      if (newCaption) {
        setCaption(newCaption);
        console.log('Caption state updated');

        // Cancel any ongoing speech before starting new one
        window.speechSynthesis.cancel();
        
        // Small delay to ensure cancel completes
        setTimeout(() => {
          const utterance = new SpeechSynthesisUtterance(newCaption);
          utterance.rate = 1.0;
          utterance.pitch = 1.0;
          utterance.volume = 1.0;
          window.speechSynthesis.speak(utterance);
          console.log('Speech started for:', newCaption);
        }, 100);
      } else {
        console.log('No caption in response');
      }
    } catch (err) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        console.log('Request cancelled');
        return;
      }
      console.error('Error details:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      });
      setError(err.response?.data?.error || err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
      isProcessingRef.current = false;
    }
  };

  // Handle file upload
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

  // Webcam capture interval (only for webcam mode)
  useEffect(() => {
    if (mode !== 'webcam') {
      return;
    }

    console.log('useEffect started - interval will run every 3 seconds');
    
    const interval = setInterval(async () => {
      console.log('Interval tick - checking conditions...');
      console.log('isProcessing:', isProcessingRef.current);
      console.log('webcamRef.current:', webcamRef.current);
      
      // Skip if already processing to prevent overlapping requests
      if (isProcessingRef.current || !webcamRef.current) {
        console.log('Skipping this interval');
        return;
      }

      const imageSrc = webcamRef.current.getScreenshot();
      console.log('Screenshot captured:', imageSrc ? 'Yes' : 'No');
      
      if (!imageSrc) {
        setError('Failed to capture image');
        return;
      }

      await processImage(imageSrc);
    }, 3000); // Increased to 3 seconds for better performance

    // Cleanup function
    return () => {
      clearInterval(interval);
      window.speechSynthesis.cancel();
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [mode]); // Run when mode changes

  return (
    <div style={{ 
      textAlign: 'center', 
      padding: '20px',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <h1>Image Captioning</h1>
      
      {/* Mode Toggle */}
      <div style={{ 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'center',
        gap: '10px'
      }}>
        <button
          onClick={() => {
            setMode('webcam');
            setUploadedImage(null);
            setCaption('');
            setError('');
            setLastProcessedImage(null);
          }}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            border: '2px solid',
            borderColor: mode === 'webcam' ? '#3b82f6' : '#e5e7eb',
            backgroundColor: mode === 'webcam' ? '#3b82f6' : 'white',
            color: mode === 'webcam' ? 'white' : '#1f2937',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: mode === 'webcam' ? '600' : '400',
            transition: 'all 0.2s'
          }}
        >
          📷 Webcam
        </button>
        <button
          onClick={() => {
            setMode('upload');
            setCaption('');
            setError('');
            setLastProcessedImage(null);
          }}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            border: '2px solid',
            borderColor: mode === 'upload' ? '#3b82f6' : '#e5e7eb',
            backgroundColor: mode === 'upload' ? '#3b82f6' : 'white',
            color: mode === 'upload' ? 'white' : '#1f2937',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: mode === 'upload' ? '600' : '400',
            transition: 'all 0.2s'
          }}
        >
          📁 Upload Photo
        </button>
      </div>

      {/* Image Display Area */}
      <div style={{ 
        display: 'inline-block', 
        position: 'relative',
        borderRadius: '8px',
        overflow: 'hidden',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        {mode === 'webcam' ? (
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={640}
            height={480}
            onUserMediaError={(err) => setError('Camera access denied')}
          />
        ) : (
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
        )}
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

      {/* File Upload Button (only in upload mode) */}
      {mode === 'upload' && (
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
      )}

      {/* Caption Display */}
      <div style={{ marginTop: '20px' }}>
        <h2 style={{ 
          color: error ? '#dc2626' : '#1f2937',
          minHeight: '40px'
        }}>
          {error ? `⚠️ ${error}` : caption ? `📷 ${caption}` : mode === 'webcam' ? '👁️ Watching...' : '📁 Select a photo to generate caption'}
        </h2>
      </div>

      {/* Try again: re-send same photo for another inference */}
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

      {mode === 'webcam' && (
        <p style={{ 
          color: '#6b7280', 
          fontSize: '14px',
          marginTop: '10px' 
        }}>
          Captions update every 3 seconds
        </p>
      )}
    </div>
  );
}

export default App;
