import {
  useEffect,
  useRef,
  useImperativeHandle,
  forwardRef,
} from 'react';

/**
 * YouTube Player component using the IFrame Player API.
 * Provides programmatic control over video playback.
 *
 * @param {Object} props
 * @param {string} props.videoId - YouTube video ID
 * @param {Function} props.onReady - Called when player is ready
 * @param {Function} props.onTimeUpdate - Called periodically with current time
 * @param {string} props.className - Optional class for outer container styling
 * @param {Object} ref - Forwarded ref exposing player controls
 */
const YouTubePlayer = forwardRef(
  ({ videoId, onReady, onTimeUpdate, className = '' }, ref) => {
    const containerRef = useRef(null);
    const playerRef = useRef(null);
    const intervalRef = useRef(null);

    // Expose player controls to parent via ref
  useImperativeHandle(ref, () => ({
    seekTo: (seconds) => {
      if (playerRef.current && playerRef.current.seekTo) {
        playerRef.current.seekTo(seconds, true);
      }
    },
    playVideo: () => {
      if (playerRef.current && playerRef.current.playVideo) {
        playerRef.current.playVideo();
      }
    },
    pauseVideo: () => {
      if (playerRef.current && playerRef.current.pauseVideo) {
        playerRef.current.pauseVideo();
      }
    },
    getCurrentTime: () => {
      if (playerRef.current && playerRef.current.getCurrentTime) {
        return playerRef.current.getCurrentTime();
      }
      return 0;
    },
  }));

    useEffect(() => {
      // Load YouTube IFrame API script
      if (!window.YT) {
        const tag = document.createElement('script');
        tag.src = 'https://www.youtube.com/iframe_api';
        const firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }

    // Initialize player when API is ready
    const initPlayer = () => {
      if (window.YT && window.YT.Player) {
        playerRef.current = new window.YT.Player(containerRef.current, {
          videoId: videoId,
          playerVars: {
            autoplay: 0,
            modestbranding: 1,
            rel: 0,
          },
          events: {
            onReady: (event) => {
              if (onReady) onReady(event);

              // Set up time update interval
              if (onTimeUpdate) {
                intervalRef.current = setInterval(() => {
                  if (playerRef.current && playerRef.current.getCurrentTime) {
                    const currentTime = playerRef.current.getCurrentTime();
                    onTimeUpdate(currentTime);
                  }
                }, 500); // Update every 500ms
              }
            },
            onStateChange: (event) => {
              // Can add state change handling here if needed
            },
          },
        });
      }
    };

      if (window.YT && window.YT.Player) {
        initPlayer();
      } else {
        window.onYouTubeIframeAPIReady = initPlayer;
      }

      // Cleanup
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
        if (playerRef.current && playerRef.current.destroy) {
          playerRef.current.destroy();
        }
      };
    }, [videoId, onReady, onTimeUpdate]);

    const outerClass = ['relative w-full overflow-hidden', className]
      .filter(Boolean)
      .join(' ');

    return (
      <div className={outerClass} style={{ aspectRatio: '16 / 9' }}>
        <div ref={containerRef} className="absolute inset-0 h-full w-full" />
      </div>
    );
  },
);

YouTubePlayer.displayName = 'YouTubePlayer';

export default YouTubePlayer;
