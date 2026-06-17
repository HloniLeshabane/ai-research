import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';

export function CodeEditor({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <CodeMirror
      value={value}
      onChange={onChange}
      theme="dark"
      extensions={[python()]}
      height="100%"
      style={{ height: '100%', fontSize: 13 }}
      basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: true }}
    />
  );
}
