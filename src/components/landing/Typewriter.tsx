"use client";

import { useState, useEffect } from "react";

const PHRASES = [
  "Ask natural language questions to complex Earth Observation data.",
  "Analyze multi-sensor optical and SAR imagery in real-time.",
  "Generate precise land-cover bounding boxes instantly."
];

export function Typewriter() {
  const [text, setText] = useState("");
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    
    const currentPhrase = PHRASES[phraseIndex];
    
    if (isDeleting) {
      if (text.length > 0) {
        timer = setTimeout(() => {
          setText(currentPhrase.substring(0, text.length - 1));
        }, 30);
      } else {
        timer = setTimeout(() => {
          setIsDeleting(false);
          setPhraseIndex((prev) => (prev + 1) % PHRASES.length);
        }, 400);
      }
    } else {
      if (text.length < currentPhrase.length) {
        timer = setTimeout(() => {
          setText(currentPhrase.substring(0, text.length + 1));
        }, 60);
      } else {
        timer = setTimeout(() => {
          setIsDeleting(true);
        }, 2500);
      }
    }

    return () => clearTimeout(timer);
  }, [text, isDeleting, phraseIndex]);

  return (
    <div className="mt-8 flex h-7 items-center justify-center text-center">
      <p className="font-mono text-[15px] md:text-[17px] font-medium tracking-wide text-white">
        {text}
        <span className="ml-1 inline-block h-[1em] w-[2px] bg-white align-middle animate-pulse"></span>
      </p>
    </div>
  );
}
