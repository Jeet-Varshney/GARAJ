import { useState, useEffect, useRef, useCallback } from 'react';

const getDefaultWsUrl = () => {
  const apiUrl = import.meta.env.VITE_API_URL;

  if (!apiUrl) {
    console.error('[WS] VITE_API_URL is not configured');
    return '';
  }

  return `${apiUrl.replace(/^http/, 'ws')}/ws/stream`;
};

export function useWebSocket(url = getDefaultWsUrl()) {
  const [connectionStatus, setConnectionStatus] = useState('DISCONNECTED');
  const [latestTelemetry, setLatestTelemetry] = useState(null);
  const [roundTripLatency, setRoundTripLatency] = useState(0);
  const [sentChunksCount, setSentChunksCount] = useState(0);
  
  const wsRef = useRef(null);
  const sequenceIdRef = useRef(0);
  const isManuallyClosedRef = useRef(false);

  const clearTelemetry = useCallback(() => {
    setLatestTelemetry(null);
    setRoundTripLatency(0);
    setSentChunksCount(0);
    sequenceIdRef.current = 0;
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setConnectionStatus('CONNECTING');
    isManuallyClosedRef.current = false;

    try {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('[WS] connected');
        setConnectionStatus('CONNECTED');
        sequenceIdRef.current = 0;
        setSentChunksCount(0);
      };

      ws.onmessage = (event) => {
        try {
          console.log('[WS RAW MESSAGE]', event.data);
          const data = JSON.parse(event.data);
          console.log('[WS PARSED]', data);
          
          if (data.timestamps && data.timestamps.client_ts_ms) {
            const clientSendTs = data.timestamps.client_ts_ms;
            const rtt = Math.max(0, Date.now() - clientSendTs);
            setRoundTripLatency(rtt);
          }

          setLatestTelemetry(data);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('ERROR');
      };

      ws.onclose = () => {
        setConnectionStatus('DISCONNECTED');
        if (!isManuallyClosedRef.current) {
          setTimeout(() => {
            if (!isManuallyClosedRef.current) {
              connect();
            }
          }, 2000);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionStatus('ERROR');
    }
  }, [url]);

  const disconnect = useCallback(() => {
    isManuallyClosedRef.current = true;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('DISCONNECTED');
    clearTelemetry();
  }, [clearTelemetry]);

  const sendAudioChunk = useCallback((pcmArrayBuffer) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    const seqId = sequenceIdRef.current++;
    const clientTsMs = Date.now();

    const headerBuffer = new ArrayBuffer(12);
    const headerView = new DataView(headerBuffer);
    headerView.setUint32(0, seqId, true);
    headerView.setFloat64(4, clientTsMs, true);

    const combinedBuffer = new Uint8Array(headerBuffer.byteLength + pcmArrayBuffer.byteLength);
    combinedBuffer.set(new Uint8Array(headerBuffer), 0);
    combinedBuffer.set(new Uint8Array(pcmArrayBuffer), headerBuffer.byteLength);

    wsRef.current.send(combinedBuffer.buffer);
    const nextCount = seqId + 1;
    setSentChunksCount(nextCount);
    console.log('[WS TX] chunks increasing', nextCount);
    return true;
  }, []);

  useEffect(() => {
    return () => {
      isManuallyClosedRef.current = true;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    connectionStatus,
    latestTelemetry,
    roundTripLatency,
    sentChunksCount,
    connect,
    disconnect,
    sendAudioChunk,
    clearTelemetry,
  };
}
