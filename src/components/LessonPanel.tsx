import type { Lesson } from '../modules/types';

export function LessonPanel({ lesson }: { lesson: Lesson }) {
  return (
    <div className="max-h-[38vh] overflow-y-auto border-b border-neutral-800 bg-neutral-900/40 px-5 py-3">
      <p className="text-sm leading-relaxed text-neutral-300">{lesson.concept}</p>
      {lesson.bridge && (
        <p className="mt-1.5 text-sm leading-relaxed text-emerald-300/90">
          <span className="text-emerald-500">EE bridge — </span>
          {lesson.bridge}
        </p>
      )}
      {lesson.tryThis && lesson.tryThis.length > 0 && (
        <ul className="mt-2 list-inside list-disc text-[13px] text-neutral-400">
          {lesson.tryThis.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
