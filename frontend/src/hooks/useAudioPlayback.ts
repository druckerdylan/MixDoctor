/**
 * Playback for the timeline transport.
 *
 * Deliberately a plain HTMLAudioElement — no WebAudio graph. The only thing we
 * add is a requestAnimationFrame clock, because the native `timeupdate` event
 * fires roughly four times a second and a playhead driven off it visibly
 * staircases across the waveform.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface AudioPlayback {
  isPlaying: boolean;
  /** Seconds. Updated every animation frame while playing. */
  currentTime: number;
  /** Seconds. 0 until metadata has loaded. */
  duration: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (seconds: number) => void;
  /** True only when there is a decodable source with known duration. */
  ready: boolean;
  error: string | null;
}

export function useAudioPlayback(audioUrl: string | null): AudioPlayback {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopClock = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const startClock = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    const tick = (): void => {
      const el = audioRef.current;
      if (!el) {
        rafRef.current = null;
        return;
      }
      setCurrentTime(el.currentTime);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    stopClock();
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setError(null);

    // No local file — e.g. a restored history entry. Report not-ready and let
    // the UI hide the transport rather than render a dead play button.
    if (!audioUrl) {
      audioRef.current = null;
      setReady(false);
      return;
    }

    const el = new Audio();
    el.preload = 'metadata';
    el.crossOrigin = 'anonymous';
    audioRef.current = el;
    setReady(false);

    const readDuration = (): void => {
      if (Number.isFinite(el.duration) && el.duration > 0) setDuration(el.duration);
    };
    const onLoadedMetadata = (): void => {
      readDuration();
      setReady(true);
    };
    const onDurationChange = (): void => readDuration();
    const onTimeUpdate = (): void => setCurrentTime(el.currentTime);
    const onSeeked = (): void => setCurrentTime(el.currentTime);
    const onPlay = (): void => {
      setIsPlaying(true);
      startClock();
    };
    const onPause = (): void => {
      setIsPlaying(false);
      stopClock();
      setCurrentTime(el.currentTime);
    };
    const onEnded = (): void => {
      setIsPlaying(false);
      stopClock();
      setCurrentTime(Number.isFinite(el.duration) ? el.duration : 0);
    };
    const onError = (): void => {
      stopClock();
      setIsPlaying(false);
      setReady(false);
      setError('This file could not be decoded for playback in the browser.');
    };

    el.addEventListener('loadedmetadata', onLoadedMetadata);
    el.addEventListener('durationchange', onDurationChange);
    el.addEventListener('timeupdate', onTimeUpdate);
    el.addEventListener('seeked', onSeeked);
    el.addEventListener('play', onPlay);
    el.addEventListener('playing', onPlay);
    el.addEventListener('pause', onPause);
    el.addEventListener('ended', onEnded);
    el.addEventListener('error', onError);

    el.src = audioUrl;
    el.load();

    return () => {
      stopClock();
      el.removeEventListener('loadedmetadata', onLoadedMetadata);
      el.removeEventListener('durationchange', onDurationChange);
      el.removeEventListener('timeupdate', onTimeUpdate);
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('play', onPlay);
      el.removeEventListener('playing', onPlay);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('ended', onEnded);
      el.removeEventListener('error', onError);
      el.pause();
      el.removeAttribute('src');
      // Release the decoder/network handle. Listeners are already detached, so
      // the abort this provokes cannot surface as a user-visible error.
      el.load();
      if (audioRef.current === el) audioRef.current = null;
    };
  }, [audioUrl, startClock, stopClock]);

  const play = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    const attempt = el.play();
    if (attempt && typeof attempt.catch === 'function') {
      attempt.catch((reason: unknown) => {
        // A pause() landing mid-promise rejects with AbortError; that is not a
        // failure the producer needs to hear about.
        const name = reason instanceof DOMException ? reason.name : '';
        if (name === 'AbortError') return;
        setIsPlaying(false);
        setError(
          name === 'NotAllowedError'
            ? 'The browser blocked playback. Press play again.'
            : 'Playback failed to start.',
        );
      });
    }
  }, []);

  const pause = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  const toggle = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) play();
    else el.pause();
  }, [play]);

  const seek = useCallback((seconds: number) => {
    const el = audioRef.current;
    if (!el) return;
    const known = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : null;
    const clamped = Math.max(0, known === null ? seconds : Math.min(seconds, known));
    if (!Number.isFinite(clamped)) return;
    try {
      el.currentTime = clamped;
    } catch {
      // Seeking before the media is seekable throws in some browsers.
      return;
    }
    setCurrentTime(clamped);
  }, []);

  // Belt and braces: the effect cleanup already cancels, but an unmount during
  // an in-flight frame should never leave a callback pointing at dead state.
  useEffect(() => stopClock, [stopClock]);

  return { isPlaying, currentTime, duration, play, pause, toggle, seek, ready, error };
}

export default useAudioPlayback;
