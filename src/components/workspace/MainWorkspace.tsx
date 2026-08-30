import { Menu, MoreHorizontal, BookOpen, Brain, CheckCircle2 } from "lucide-react";

export function MainWorkspace({ onMenuClick }: { onMenuClick: () => void }) {
  return (
    <main className="flex min-w-0 flex-1 flex-col bg-sat-bg">
      {/* Top Header / Metadata */}
      <header className="flex h-14 items-center justify-between border-b border-sat-hairline px-4 md:px-8">
        <div className="flex items-center gap-3">
          <button onClick={onMenuClick} className="mr-2 text-sat-nav md:hidden">
            <Menu size={20} />
          </button>
          <div className="flex items-center gap-2 rounded-full border border-sat-hairline bg-sat-box px-3 py-1 text-xs text-sat-subtitle">
            <span className="font-medium text-sat-heading">Math</span>
            <span className="h-3 w-px bg-sat-hairline" />
            <span>Heart of Algebra</span>
            <span className="h-3 w-px bg-sat-hairline" />
            <span>Hard</span>
          </div>
        </div>
        <button className="text-sat-nav hover:text-sat-heading">
          <MoreHorizontal size={20} />
        </button>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-8 md:px-8 md:py-12">
          
          {/* Question Section */}
          <section className="mb-12">
            <h1 className="mb-6 text-xl font-medium text-sat-heading md:text-2xl">
              Question 14
            </h1>
            
            {/* The Question Image representation */}
            <div className="mb-8 overflow-hidden rounded-xl border border-sat-hairline bg-sat-box p-6 shadow-sm">
              <p className="mb-6 text-[15px] leading-relaxed text-sat-msg-text">
                If <span className="font-mono text-sat-heading text-sm">3x - y = 12</span> and <span className="font-mono text-sat-heading text-sm">y = 3/2</span>, what is the value of <span className="font-mono text-sat-heading text-sm">10x</span>?
              </p>
              
              <div className="space-y-3">
                {["A) 15", "B) 30", "C) 45", "D) 60"].map((opt, i) => (
                  <div key={i} className={`flex items-center gap-3 rounded-lg border ${i === 2 ? "border-sat-green/30 bg-sat-green/5" : "border-sat-hairline bg-sat-bg"} p-4`}>
                    <div className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-medium ${i === 2 ? "bg-sat-green text-black" : "border border-sat-hairline text-sat-subtitle"}`}>
                      {opt[0]}
                    </div>
                    <span className="text-[15px] text-sat-msg-text">{opt.substring(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Solution Section */}
          <section className="mb-12">
            <div className="mb-6 flex items-center gap-2">
              <BookOpen size={18} className="text-sat-subtitle" />
              <h2 className="text-lg font-medium text-sat-heading">Written Solution</h2>
            </div>
            
            <div className="prose prose-invert max-w-none text-[15px] leading-relaxed text-sat-nav">
              <p className="mb-4">
                To find the value of <span className="font-mono text-sat-heading text-sm">10x</span>, we first need to solve for <span className="font-mono text-sat-heading text-sm">x</span> using the given system of equations.
              </p>
              
              <div className="my-6 space-y-4 rounded-xl border border-sat-hairline bg-sat-box p-6">
                <div className="flex gap-4">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sat-hairline text-xs font-medium text-sat-heading">1</span>
                  <div>
                    <p className="mb-2 text-sat-msg-text">Substitute the value of y into the first equation:</p>
                    <p className="font-mono text-sat-heading text-sm">3x - (3/2) = 12</p>
                  </div>
                </div>
                
                <div className="flex gap-4">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sat-hairline text-xs font-medium text-sat-heading">2</span>
                  <div>
                    <p className="mb-2 text-sat-msg-text">Add 3/2 to both sides to isolate the x term:</p>
                    <p className="font-mono text-sat-heading text-sm">3x = 12 + 1.5</p>
                    <p className="mt-1 font-mono text-sat-heading text-sm">3x = 13.5</p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sat-hairline text-xs font-medium text-sat-heading">3</span>
                  <div>
                    <p className="mb-2 text-sat-msg-text">Divide by 3 to find x:</p>
                    <p className="font-mono text-sat-heading text-sm">x = 4.5</p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sat-hairline text-xs font-medium text-sat-heading">4</span>
                  <div>
                    <p className="mb-2 text-sat-msg-text">Multiply by 10 to get the final answer:</p>
                    <p className="font-mono text-sat-green text-sm font-semibold">10x = 10(4.5) = 45</p>
                  </div>
                </div>
              </div>
              
              <p>
                Therefore, the correct answer is <strong className="text-sat-heading font-medium">C) 45</strong>.
              </p>
            </div>
          </section>

          {/* Additional Details */}
          <section className="border-t border-sat-hairline pt-8">
            <h3 className="mb-6 text-sm font-medium tracking-wide text-sat-nav uppercase">
              Additional Details
            </h3>
            
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-sat-hairline bg-[#0c0c0c] p-5">
                <div className="mb-2 flex items-center gap-2 text-sat-heading">
                  <Brain size={16} />
                  <h4 className="font-medium text-[14px]">Skill Tested</h4>
                </div>
                <p className="text-[13px] leading-relaxed text-sat-subtitle">
                  Solving linear equations and interpreting the solution in the context of an expression.
                </p>
              </div>
              
              <div className="rounded-xl border border-sat-hairline bg-[#0c0c0c] p-5">
                <div className="mb-2 flex items-center gap-2 text-sat-heading">
                  <CheckCircle2 size={16} />
                  <h4 className="font-medium text-[14px]">Key Takeaway</h4>
                </div>
                <p className="text-[13px] leading-relaxed text-sat-subtitle">
                  Always check what the question is asking for. Here, it asks for 10x, not just x.
                </p>
              </div>
            </div>
          </section>
          
        </div>
      </div>
    </main>
  );
}

