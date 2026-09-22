// Build the one-page handout (demo/handout/the-missing-band.docx) from handout.json.
// usage: npm i docx && node build.js handout.json the-missing-band.docx 0.9
// Every number in handout.json is from web/public/assets/results.json (run OV50); change the JSON, not the docx.
const fs = require('fs');
const {
  AlignmentType, BorderStyle, Document, HeadingLevel, LevelFormat, Packer, PageNumber, Paragraph, ShadingType,
  Table, TableCell, TableRow, TextRun, WidthType, VerticalAlign, TabStopType, Footer, LineRuleType,
} = require('docx');

const [,, inPath, outPath, scaleArg] = process.argv;
const D = JSON.parse(fs.readFileSync(inPath, 'utf8'));
const S = Number(scaleArg || 1);          // global type scale to fit one page
const pt = (v) => Math.round(v * S * 2);  // half-points

// Fonts: Arial for Latin (metric-compatible with Liberation Sans, which renders here), Yu Gothic for
// kana and kanji (ships with Office on Windows and Mac; LibreOffice substitutes Noto Sans CJK).
const FONT = { ascii: 'Arial', hAnsi: 'Arial', cs: 'Arial', eastAsia: 'Yu Gothic' };
const MONO = { ascii: 'Consolas', hAnsi: 'Consolas', cs: 'Consolas', eastAsia: 'Yu Gothic' };
const INK = '111318', DIM = '5A6270', LINE = 'C9CED6', WARM = 'B86E00', PANEL = 'F3F4F6';

const run = (text, o = {}) => new TextRun({ text, font: o.mono ? MONO : FONT, size: pt(o.size ?? 9.5), bold: o.bold, italics: o.italics, color: o.color ?? INK });

// "**bold**" and "`mono`" inline marks, kept minimal
function rich(text, o = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(run(text.slice(last, m.index), o));
    const tok = m[0];
    if (tok.startsWith('**')) out.push(run(tok.slice(2, -2), { ...o, bold: true, color: o.mono ? INK : (o.color ?? INK) }));
    else out.push(run(tok.slice(1, -1), { ...o, mono: true, size: (o.size ?? 9.5) - 0.5 }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(run(text.slice(last), o));
  return out;
}

const para = (children, o = {}) => new Paragraph({ children, spacing: { before: o.before ?? 0, after: o.after ?? 60, line: o.line ?? 252 }, alignment: o.align, numbering: o.numbering, border: o.border, keepNext: o.keepNext, keepLines: true });

const head = (text) => para([run(text.toUpperCase(), { size: 8.5, bold: true, color: WARM })], { before: 110, after: 36, keepNext: true, line: 240 });

const spacer = (h) => new Paragraph({ children: [], spacing: { before: 0, after: 0, line: h, lineRule: LineRuleType.EXACT } });
const rule = () => new Paragraph({ children: [], spacing: { before: 0, after: 80 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LINE, space: 1 } } });

const noBorder = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const thin = { style: BorderStyle.SINGLE, size: 4, color: LINE };

function abstractBox(text) {
  return new Table({
    width: { size: 9860, type: WidthType.DXA }, columnWidths: [9860],
    borders: { top: noBorder, bottom: noBorder, left: { style: BorderStyle.SINGLE, size: 18, color: WARM }, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
    rows: [new TableRow({ children: [new TableCell({
      width: { size: 9860, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: PANEL, color: 'auto' },
      margins: { top: 90, bottom: 90, left: 160, right: 160 },
      children: [
        para([run('要旨', { size: 8.5, bold: true, color: DIM })], { after: 30, line: 240 }),
        para([run(text, { size: 8.7 })], { after: 0, line: 290 }),
      ],
    })] })],
  });
}


const GOOD = '1E7A46';
function tiles(TS) {
  const n = TS.length, total = 9860, gap = 120;
  const w = Math.floor((total - gap * (n - 1)) / n);
  const widths = []; for (let i = 0; i < n; i++) { widths.push(w); if (i < n - 1) widths.push(gap); }
  const sum = widths.reduce((a, b) => a + b, 0); widths[widths.length - 1] += total - sum;
  const cells = [];
  TS.forEach((tile, i) => {
    cells.push(new TableCell({
      width: { size: widths[2 * i], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: PANEL, color: 'auto' },
      borders: { top: { style: BorderStyle.SINGLE, size: 18, color: WARM }, bottom: noBorder, left: noBorder, right: noBorder },
      margins: { top: 70, bottom: 80, left: 140, right: 100 },
      children: [
        para([run(tile.label, { size: 7.5, color: DIM })], { after: 0, line: 220 }),
        para([run(tile.value, { size: 21, bold: true })], { before: 40, after: 20, line: 300 }),
        para([run(tile.delta, { size: 8, bold: tile.good != null, color: tile.good === true ? GOOD : tile.good === false ? 'B3261E' : DIM })], { after: 0, line: 220 }),
      ],
    }));
    if (i < n - 1) cells.push(new TableCell({ width: { size: gap, type: WidthType.DXA }, borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder }, children: [para([], { after: 0 })] }));
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
    rows: [new TableRow({ children: cells })] });
}

function table(T) {
  const n = T.columns.length;
  const total = 9860;
  const widths = n === 2 ? [4500, 5360] : n === 3 ? [3160, 3350, 3350] : [3000, ...Array(n - 1).fill(Math.floor((total - 3000) / (n - 1)))];
  const sum = widths.reduce((a, b) => a + b, 0); widths[widths.length - 1] += total - sum;
  const cell = (text, i, hdr) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: hdr ? { type: ShadingType.CLEAR, fill: PANEL, color: 'auto' } : (i === 2 ? { type: ShadingType.CLEAR, fill: 'FFF7EA', color: 'auto' } : undefined),
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [para(rich(text, { size: !hdr && i === 2 ? 9.5 : 8.5, bold: hdr, color: hdr ? DIM : INK, mono: !hdr && i > 0 }), { after: 0, line: 230, align: i > 0 ? AlignmentType.RIGHT : AlignmentType.LEFT })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: thin, bottom: thin, left: noBorder, right: noBorder, insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'E3E6EB' }, insideVertical: noBorder },
    rows: [
      new TableRow({ tableHeader: true, children: T.columns.map((c, i) => cell(c, i, true)) }),
      ...T.rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

const children = [];
children.push(para([run(D.title, { size: 17, bold: true })], { after: 20, line: 240 }));
children.push(para([run(D.subtitle, { size: 9.5, color: DIM })], { after: 90, line: 240 }));
children.push(rule());
if (D.tiles?.length) { children.push(tiles(D.tiles)); children.push(spacer(140)); }
children.push(abstractBox(D.abstract_ja));
children.push(spacer(80));

for (const sec of D.sections) {
  children.push(head(sec.heading));
  for (const p of sec.paragraphs) children.push(para(rich(p), { after: 60 }));
  for (const b of sec.bullets) children.push(para(rich(b, { size: 9 }), { numbering: { reference: 'bul', level: 0 }, after: 30, line: 240 }));
  if (sec.table_after && D.table) {
    children.push(para([run(D.table.caption, { size: 8, color: DIM, italics: true })], { before: 60, after: 40, line: 240, keepNext: true }));
    children.push(table(D.table));
    children.push(spacer(100));
    D.table = null;
  }
}
if (D.table) {
  children.push(para([run(D.table.caption, { size: 8, color: DIM, italics: true })], { before: 100, after: 40, line: 240, keepNext: true }));
  children.push(table(D.table));
}
if (D.caveats?.length) {
  children.push(head(D.caveats_heading || 'What this is not'));
  for (const b of D.caveats) children.push(para(rich(b, { size: 9 }), { numbering: { reference: 'bul', level: 0 }, after: 30, line: 240 }));
}
if (D.references?.length) {
  children.push(para([run('References  ', { size: 7.5, bold: true, color: WARM }), ...rich(D.references.join('  ·  '), { size: 7.5, color: DIM })], { before: 90, after: 0, line: 220 }));
}

const doc = new Document({
  creator: 'lisa-rtm', title: D.title, description: D.subtitle,
  styles: { default: { document: { run: { font: FONT, size: pt(9.5), color: INK } } } },
  numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 300, hanging: 200 } }, run: { color: WARM } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 760, bottom: 760, left: 1023, right: 1023, footer: 400 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: 9860 }],
      children: [run(D.footer_left || '', { size: 7.5, color: DIM }), new TextRun({ text: '\t', font: FONT, size: pt(7.5) }), run(D.footer_right || '', { size: 7.5, color: DIM })],
      spacing: { before: 0, after: 0 },
    })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log('wrote', outPath, b.length, 'bytes'); });
