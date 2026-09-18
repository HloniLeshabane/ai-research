import { useState } from 'react';
import { PyodideProvider } from './pyodide/PyodideProvider';
import { LabView } from './components/LabView';
import { ModuleNav } from './components/ModuleNav';
import { LessonPanel } from './components/LessonPanel';
import { CURRICULUM, READY_ENTRIES } from './modules/curriculum';

export default function App() {
  const [activeId, setActiveId] = useState(READY_ENTRIES[0].id);
  const active = READY_ENTRIES.find((e) => e.id === activeId) ?? READY_ENTRIES[0];
  const module = active.module;

  return (
    <PyodideProvider>
      <div className="flex h-screen bg-neutral-950 text-neutral-100">
        <ModuleNav sections={CURRICULUM} activeId={activeId} onSelect={setActiveId} />
        <main className="flex min-w-0 flex-1 flex-col">
          <header className="border-b border-neutral-800 px-5 py-3">
            <div className="font-mono text-[11px] uppercase tracking-wider text-neutral-500">
              {module.section}
            </div>
            <h1 className="font-mono text-lg">{module.title}</h1>
          </header>
          <LessonPanel lesson={module.lesson} />
          <div className="min-h-0 flex-1">
            <LabView key={module.id} module={module} />
          </div>
        </main>
      </div>
    </PyodideProvider>
  );
}
