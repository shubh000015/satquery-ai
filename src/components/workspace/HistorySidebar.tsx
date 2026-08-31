import { useState } from "react";
import { Search, Map, Layers, FileText, ChevronRight, ChevronLeft, X } from "lucide-react";
import { useSatQuery } from "@/lib/store";

export function HistorySidebar({ 
  isOpen, 
  onClose,
  isDesktopCollapsed,
  onToggleDesktop
}: { 
  isOpen: boolean; 
  onClose: () => void;
  isDesktopCollapsed?: boolean;
  onToggleDesktop?: () => void;
}) {
  const s = useSatQuery();

  if (isDesktopCollapsed) {
    return (
      <aside className="hidden flex-col border-r border-sat-hairline bg-sat-box md:flex md:w-14 transition-all duration-300">
        <header className="flex h-14 flex-col items-center justify-center gap-2 border-b border-sat-hairline py-2">
          <button onClick={onToggleDesktop} className="text-sat-nav hover:text-sat-heading" title="Expand History">
            <ChevronRight size={20} />
          </button>
        </header>
        <div className="flex-1 py-4 flex flex-col items-center gap-4">
          <button onClick={s.startNewChat} className="text-sat-nav hover:text-sat-heading pb-2" title="New Chat">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          </button>
          {s.history.map((h) => (
            <button key={h.id} onClick={() => { onToggleDesktop?.(); s.loadSession(h.id); }} className="text-sat-nav hover:text-sat-heading" title={h.title}>
              <Map size={18} className={s.activeSessionId === h.id ? "text-sat-heading" : ""} />
            </button>
          ))}
        </div>
      </aside>
    );
  }

  const [openMenuId, setOpenMenuId] = useState<string | null>(null);

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 md:hidden" 
          onClick={onClose} 
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col border-r border-sat-hairline bg-sat-box transition-transform duration-300 md:static md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        onClick={() => setOpenMenuId(null)}
      >
        <header className="flex h-14 items-center justify-between border-b border-sat-hairline px-4">
          <button
            type="button"
            onClick={s.goIngress}
            className="text-sm font-semibold tracking-wide text-sat-heading"
          >
            SatQuery AI
          </button>
          <div className="flex items-center gap-3">
            <button 
              onClick={s.startNewChat} 
              className="text-sat-nav hover:text-sat-heading transition-colors" 
              title="New Chat"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
            </button>
            {onToggleDesktop && (
              <button onClick={onToggleDesktop} className="hidden text-sat-nav hover:text-sat-heading md:block" title="Collapse History">
                <ChevronLeft size={20} />
              </button>
            )}
            <button onClick={onClose} className="text-sat-nav md:hidden">
              <X size={18} />
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-3 py-4">
          <div className="mb-6">
            <h3 className="mb-3 px-2 text-[11px] font-medium tracking-wider text-sat-nav uppercase">
              Recent Chats
            </h3>
            {s.history.length === 0 ? (
              <p className="px-2 text-[12px] text-sat-nav">No chats yet.</p>
            ) : (
              <ul className="space-y-1">
                {s.history.map((item) => (
                  <li key={item.id} className="group relative flex items-center">
                    <button
                      onClick={() => s.loadSession(item.id)}
                      className={`w-full flex items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors pr-8 ${
                        s.activeSessionId === item.id 
                          ? "bg-sat-hairline text-sat-heading" 
                          : "text-sat-nav hover:bg-sat-hairline/50 hover:text-sat-heading"
                      }`}
                    >
                      <Map size={16} className={s.activeSessionId === item.id ? "text-sat-heading" : "text-sat-placeholder"} />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[13px] font-medium">{item.title}</p>
                        <p className="truncate text-[11px] text-sat-subtitle">{new Date(item.date).toLocaleDateString()}</p>
                      </div>
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuId(openMenuId === item.id ? null : item.id);
                      }}
                      className="absolute right-2 hidden p-1 text-sat-nav hover:text-sat-heading group-hover:block"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="1"/><circle cx="12" cy="5" r="1"/><circle cx="12" cy="19" r="1"/></svg>
                    </button>

                    {openMenuId === item.id && (
                      <div className="absolute right-2 top-8 z-50 w-28 rounded-md border border-sat-hairline bg-[#1c1c1c] shadow-lg">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            s.deleteSession(item.id);
                            setOpenMenuId(null);
                          }}
                          className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-[12px] text-red-400 hover:bg-white/5"
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/></svg>
                          Delete
                        </button>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}


