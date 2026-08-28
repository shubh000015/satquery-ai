"use client";

import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Ingress } from "@/components/ingress/Ingress";
import { Workspace } from "@/components/workspace/Workspace";
import { SatQueryProvider, useSatQuery } from "@/lib/store";

export function SatQueryApp() {
  return (
    <SatQueryProvider>
      <Shell />
    </SatQueryProvider>
  );
}

function Shell() {
  const { screen, setReportOpen, reportOpen, measuring, setMeasuring } = useSatQuery();

  useEffect(() => {
    const locked = screen === "workspace";
    document.documentElement.classList.toggle("app-locked", locked);
    document.body.classList.toggle("app-locked", locked);
    if (locked) window.scrollTo(0, 0);
    return () => {
      document.documentElement.classList.remove("app-locked");
      document.body.classList.remove("app-locked");
    };
  }, [screen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (e.key === "Escape") {
        if (reportOpen) setReportOpen(false);
        else if (measuring) setMeasuring(false);
        return;
      }
      if (e.key === "/" && tag !== "TEXTAREA" && tag !== "INPUT") {
        e.preventDefault();
        document.getElementById("SatQuery-input")?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reportOpen, measuring, setReportOpen, setMeasuring]);

  return (
    <AnimatePresence mode="wait">
      {screen === "ingress" ? (
        <motion.div
          key="in"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35 }}
        >
          <Ingress />
        </motion.div>
      ) : (
        <motion.div
          key="ws"
          className="h-dvh overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35 }}
        >
          <Workspace />
        </motion.div>
      )}
    </AnimatePresence>
  );
}


