import { PyodideProvider } from './pyodide/PyodideProvider';
import { LabView } from './components/LabView';
import { MODULES } from './modules/registry';

export default function App() {
  const module = MODULES[0];

  return (
    <PyodideProvider>
      <div className="flex h-screen flex-col bg-neutral-950 text-neutral-100">
        <header className="border-b border-neutral-800 px-5 py-3">
          <h1 className="font-mono text-lg">AI Lab — {module.title}</h1>
          <p className="font-mono text-xs text-neutral-500">{module.blurb}</p>
        </header>
        <div className="min-h-0 flex-1">
          <LabView module={module} />
        </div>
      </div>
    </PyodideProvider>
  );
}
