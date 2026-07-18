import React from 'react';

/**
 * Minimal, dependency-free Markdown renderer for chat output.
 *
 * Handles the constructs the local LLM actually produces — **bold**, *italic*,
 * `inline code`, ```code blocks```, bullet and numbered lists (incl. Persian
 * digits ۱۲۳), and headings — while rendering everything with `dir="auto"` so
 * mixed Persian / Latin / numbers keep correct bidirectional order. Builds real
 * React nodes (no dangerouslySetInnerHTML) so it is XSS-safe.
 */

// inline: **bold**, *italic*, `code`
function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    const key = `${keyPrefix}-${i++}`;
    if (tok.startsWith('**')) {
      nodes.push(<strong key={key} className="font-semibold text-white">{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith('`')) {
      nodes.push(<code key={key} className="px-1 py-0.5 rounded bg-black/40 text-[0.85em] font-mono text-indigo-200" dir="ltr">{tok.slice(1, -1)}</code>);
    } else {
      nodes.push(<em key={key} className="italic">{tok.slice(1, -1)}</em>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const BULLET = /^\s*([-*•])\s+(.*)$/;
const NUMBERED = /^\s*([0-9۰-۹]+)[.)۰-۹]*[.)]\s+(.*)$/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*)$/;

export default function Markdown({ content }: { content: string }) {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // fenced code block
    if (/^\s*```/.test(line)) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // closing fence
      blocks.push(
        <pre key={key++} dir="ltr" className="my-1.5 p-2.5 rounded-lg bg-black/50 overflow-x-auto text-[0.82em] font-mono text-gray-200 leading-relaxed">
          <code>{buf.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    // heading
    const h = HEADING.exec(line);
    if (h) {
      blocks.push(<div key={key++} dir="auto" className="font-semibold text-white mt-2 mb-0.5">{renderInline(h[2], `h${key}`)}</div>);
      i++;
      continue;
    }

    // list (bullet or numbered) — collect consecutive items
    if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = NUMBERED.test(line) && !BULLET.test(line);
      const items: React.ReactNode[] = [];
      while (i < lines.length && (BULLET.test(lines[i]) || NUMBERED.test(lines[i]))) {
        const mm = BULLET.exec(lines[i]) || NUMBERED.exec(lines[i]);
        items.push(
          <li key={key++} dir="auto" className="my-0.5">{renderInline(mm![2], `li${key}`)}</li>,
        );
        i++;
      }
      const cls = 'my-1 pr-5 space-y-0.5 ' + (ordered ? 'list-decimal' : 'list-disc');
      blocks.push(ordered
        ? <ol key={key++} dir="auto" className={cls}>{items}</ol>
        : <ul key={key++} dir="auto" className={cls}>{items}</ul>);
      continue;
    }

    // blank line -> spacing
    if (line.trim() === '') { blocks.push(<div key={key++} className="h-1.5" />); i++; continue; }

    // paragraph (merge following non-structural lines)
    const para: string[] = [line];
    i++;
    while (i < lines.length && lines[i].trim() !== '' &&
           !/^\s*```/.test(lines[i]) && !BULLET.test(lines[i]) &&
           !NUMBERED.test(lines[i]) && !HEADING.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    blocks.push(
      <div key={key++} dir="auto" className="leading-relaxed">
        {para.map((p, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && <br />}
            {renderInline(p, `p${key}-${idx}`)}
          </React.Fragment>
        ))}
      </div>,
    );
  }

  return <div className="space-y-0">{blocks}</div>;
}
