import { Search, Map, Layers, FileText, ChevronRight, X } from "lucide-react";
import { useSatQuery } from "@/lib/store";

export function HistorySidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { goIngress } = useSatQuery();

  const history = [
    { id: 1, title: "Flood Extent Analysis", location: "Patna, Bihar", active: true },
    { id: 2, title: "Urban Expansion", location: "Bengaluru South", active: false },
    { id: 3, title: "Deforestation Tracking", location: "Western Ghats", active: false },
    { id: 4, title: "Crop Health Index", location: "Punjab Region", active: false },
  ];

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
      >
        <header className="flex h-14 items-center justify-between border-b border-sat-hairline px-4">
          <button
            type="button"
            onClick={goIngress}
            className="text-sm font-semibold tracking-wide text-sat-heading"
          >
            SAT QUERY AI
          </button>
          <button onClick={onClose} className="text-sat-nav md:hidden">
            <X size={18} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-3 py-4">
          <div className="mb-6">
            <h3 className="mb-3 px-2 text-[11px] font-medium tracking-wider text-sat-nav uppercase">
              Current Session
            </h3>
            <ul className="space-y-1">
              {history.map((item) => (
                <li key={item.id}>
                  <button
                    className={`w-full flex items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors ${
                      item.active 
                        ? "bg-sat-hairline text-sat-heading" 
                        : "text-sat-nav hover:bg-sat-hairline/50 hover:text-sat-heading"
                    }`}
                  >
                    <Map size={16} className={item.active ? "text-sat-heading" : "text-sat-placeholder"} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[13px] font-medium">{item.title}</p>
                      <p className="truncate text-[11px] text-sat-subtitle">{item.location}</p>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="mb-3 px-2 text-[11px] font-medium tracking-wider text-sat-nav uppercase">
              Saved Analyses
            </h3>
            <ul className="space-y-1">
              <li>
                <button className="w-full flex items-center gap-3 rounded-lg px-2 py-2 text-left text-sat-nav transition-colors hover:bg-sat-hairline/50 hover:text-sat-heading">
                  <FileText size={16} className="text-sat-placeholder" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13px] font-medium">Coastal Erosion Report</p>
                    <p className="truncate text-[11px] text-sat-subtitle">Odisha Coast</p>
                  </div>
                </button>
              </li>
              <li>
                <button className="w-full flex items-center gap-3 rounded-lg px-2 py-2 text-left text-sat-nav transition-colors hover:bg-sat-hairline/50 hover:text-sat-heading">
                  <Layers size={16} className="text-sat-placeholder" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13px] font-medium">Water Body Detection</p>
                    <p className="truncate text-[11px] text-sat-subtitle">Chennai Region</p>
                  </div>
                </button>
              </li>
            </ul>
          </div>
        </div>
      </aside>
    </>
  );
}


