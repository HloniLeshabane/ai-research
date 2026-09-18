import type { CurriculumSection } from '../modules/curriculum';

export function ModuleNav({
  sections,
  activeId,
  onSelect,
}: {
  sections: CurriculumSection[];
  activeId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <nav className="h-full w-60 shrink-0 overflow-auto border-r border-neutral-800 bg-neutral-900 py-4">
      <div className="px-4 pb-3 font-mono text-sm text-neutral-200">AI Lab</div>
      {sections.map((section) => (
        <div key={section.title} className="mb-3">
          <div className="px-4 py-1 font-mono text-[11px] uppercase tracking-wider text-neutral-600">
            {section.title}
          </div>
          {section.entries.map((e) => {
            const ready = e.status === 'ready';
            const active = e.id === activeId;
            return (
              <button
                key={e.id}
                type="button"
                disabled={!ready}
                onClick={() => ready && onSelect(e.id)}
                className={
                  'flex w-full items-center justify-between gap-2 px-4 py-1.5 text-left font-mono text-[13px] ' +
                  (active
                    ? 'bg-emerald-600/15 text-emerald-300'
                    : ready
                      ? 'text-neutral-300 hover:bg-neutral-800'
                      : 'cursor-not-allowed text-neutral-600')
                }
              >
                <span>{e.title}</span>
                {!ready && <span className="text-[10px] uppercase text-neutral-700">soon</span>}
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}
