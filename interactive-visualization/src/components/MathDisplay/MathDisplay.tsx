import React, { useEffect, useRef, memo } from 'react';
import katex from 'katex';
import styles from './MathDisplay.module.css';

interface MathDisplayProps {
  texString: string | null | undefined; // Allow null/undefined input
  inline?: boolean;
  className?: string; // Allow passing additional classes
}

// Memoize the component to prevent unnecessary re-renders if props haven't changed
const MathDisplay: React.FC<MathDisplayProps> = memo(({ texString, inline = false, className = '' }) => {
  const containerRef = useRef<HTMLSpanElement>(null);
  const errorRef = useRef<HTMLSpanElement>(null); // Separate span for errors

  useEffect(() => {
    const container = containerRef.current;
    const errorSpan = errorRef.current;

    // Clear previous content on every render where container exists
    if (container) container.innerHTML = '';
    if (errorSpan) errorSpan.textContent = '';

    if (container && texString) { // Only render if container and string exist
      try {
        katex.render(texString, container, {
          throwOnError: true, // Use KaTeX specific errors
          displayMode: !inline,
          output: 'html',
          // Consider adding macros or trust options if needed later
        });
      } catch (error: any) {
        console.error("KaTeX rendering error:", error, "Input:", texString);
        if (errorSpan) {
           // Display a user-friendly error message
           errorSpan.textContent = `[KaTeX Error]`;
           errorSpan.title = error.message || String(error); // Put details in title
        }
         // Optionally display raw string as fallback in main container
         // container.textContent = texString;
      }
    }
  // Dependency array ensures effect runs only when texString or inline changes
  }, [texString, inline]);

  // Combine passed className with component's own class
  const combinedClassName = `${styles.mathDisplayContainer} ${className}`.trim();

  return (
    // Use spans for flexible inline/block rendering based on context
    <span className={combinedClassName}>
      <span ref={containerRef} />
      <span ref={errorRef} className={styles.katexError} />
    </span>
  );
});
// Add display name for React DevTools
MathDisplay.displayName = 'MathDisplay';
export default MathDisplay;
