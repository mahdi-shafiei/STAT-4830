import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css'; // Import KaTeX CSS
import styles from './MathDisplay.module.css';

interface MathDisplayProps {
  texString?: string; // Make optional
  displayMode?: boolean; // Render as block or inline
}

const MathDisplay: React.FC<MathDisplayProps> = ({ texString, displayMode = false }) => {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current && texString) {
      try {
        katex.render(texString, containerRef.current, {
          throwOnError: false, // Don't crash the app on KaTeX error
          displayMode: displayMode,
          output: 'html', // Ensure HTML output
        });
      } catch (error) {
        console.error('KaTeX rendering error:', error);
        // Display the original string as fallback
        if (containerRef.current) {
            containerRef.current.textContent = texString;
            containerRef.current.classList.add(styles.katexError); // Add error style
        }
      }
    } else if (containerRef.current) {
        // Clear content if texString is empty or undefined
        containerRef.current.textContent = '';
        containerRef.current.classList.remove(styles.katexError);
    }
  }, [texString, displayMode]); // Re-render if texString or displayMode changes

  // Use a span, KaTeX will handle block/inline based on displayMode
  return <span ref={containerRef} className={styles.mathContainer}></span>;
};

export default React.memo(MathDisplay); // Memoize for performance
