// Regenerate slide previews and notes from audited CSV-derived summaries.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const moduleName = process.env.CSC_ARTIFACT_MODULE;
const { Presentation } = await import(moduleName ? pathToFileURL(moduleName).href : '@oai/artifact-tool');
const facts = JSON.parse(await fs.readFile('results/controlled_v1/summary.json', 'utf8'));
const deck = Presentation.create({slideSize:{width:1280,height:720}});
const out = path.resolve('presentation');
await fs.mkdir(path.join(out,'previews'),{recursive:true});
const notes=[];
function text(s,value,x,y,w,h,size=26,color='#26384a',bold=false){
 const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 box.text=value;box.text.style={fontFamily:'Arial',fontSize:size,color,bold};
}
function slide(title,note,seconds){
 const s=deck.slides.add();s.background.fill='#ffffff';
 text(s,title,48,25,1190,66,36,'#193347',true);
 text(s,`iML 26/27 · Cognitive-Structural Complexity · ${deck.slides.items.length}`,48,690,1180,22,14,'#657585');
 s.speakerNotes.text=note;
 notes.push(`## Slide ${deck.slides.items.length} — ${title} (${seconds} seconds)\n\n${note}`);
 return s;
}
async function chart(s,name,x,y,w,h){
 s.images.add({blob:new Uint8Array(await fs.readFile(`results/controlled_v1/figures/${name}_slide.png`)),contentType:'image/png',
              fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
let s=slide('What does CSC add beyond model size?',
`CSC combines average path length, distinct split tests and distinct full trees. It is a structural proxy, not a measured cognitive cost. We preserve its formula and original study: two datasets, three seeds, twelve candidates per group, and six frozen test selections. This extension asks three questions using controlled fixtures and same-split validation comparisons. The original test data are not used to choose the new story. All 78 saved result rows were reconstructed exactly.`,40);
text(s,'CSC = mean path + log₂(1 + S) + log₂(U)',48,135,1180,65,38,'#0072B2',true);
text(s,'S: distinct split tests     U: distinct full-tree signatures',48,210,1180,42,26);
text(s,'Q1  Same size and function: does path occupancy matter?\nQ2  Same ensemble size: does distinct logic matter?\nQ3  Similar validation accuracy: do rankings differ?',48,300,1180,170,29,'#193347',true);
text(s,'Iris + Breast Cancer · 3 seeds · 12 configurations per split\n72 validation candidates · 6 frozen test choices · unchanged CSC formula',48,510,1180,90,25);
text(s,'Constructed controls + exploratory comparisons; no human validation.',48,630,1180,40,24,'#657585');
s=slide('Q1 · Equal size and predictions; different paths',
`Both constructed trees have seven nodes, four leaves and exactly the same binary prediction function. Their balanced and chain representations encounter the same three thresholds in different orders. Uniform, early-exit and deep-exit reference distributions isolate occupancy. The balanced tree has CSC four throughout. The chain scores 4.25, 3.30 and 4.85, so which representation is simpler depends on where examples fall. Node count cannot distinguish these cases. Total expected path length already detects the path difference. This establishes representation and reference dependence, not superiority to every baseline.`,55);
await chart(s,'q1_paths',40,100,1200,510);
text(s,'Same function, 7 nodes, 4 leaves · TEPL captures the path difference too.',48,630,1180,42,24,'#193347',true);
s=slide('Q2 · Fixed size; more distinct logic increases CSC',
`These artificial ensembles always contain twelve stumps and thirty-six nodes. Every example visits twelve split nodes, so total expected path length stays twelve. We vary the number of distinct stumps from one to twelve. CSC rises from two to 8.29 as distinct split and tree signatures increase. This isolates a distinction that raw size and traversal totals miss. Unlike the first control, predictive functions are not held fixed in this diversity sweep. The separate replication control averages three unique stumps repeated equally. Its probabilities and CSC remain unchanged as raw nodes and visits increase. That prediction-invariance claim applies to equal-weight averaging, not arbitrary boosting duplication.`,55);
await chart(s,'q2_diversity',40,105,1200,500);
text(s,'Diversity sweep changes functions; equal replication preserves averaging predictions.',48,630,1180,42,23,'#193347',true);
s=slide('Q3 · Near-best models mostly agree with simpler baselines',
`We enumerate all ${facts.all_pairs} unordered pairs within the same dataset and seed. ${facts.matched_pairs} pairs differ by at most two percentage points in validation balanced accuracy. For ${facts.near_best_pairs}, both models are also within two points of their group best. The displayed example is the first deterministically selected pair from each group; all eligible pairs and selection rules remain in the report. Among near-best pairs, CSC reverses node-count rankings three times and TEPL rankings seven times. One pair ties on both measures. This largely confirms agreement on this small grid. These overlapping pairs are dependent observations, not hundreds of independent experiments. Different rankings do not establish which metric better predicts human understanding.`,75);
await chart(s,'q3_pairs',35,100,1210,500);
text(s,`${facts.near_best_pairs} near-best pairs: 3 node-count reversals · 7 TEPL reversals · 1 shared tie`,48,615,1180,35,25,'#193347',true);
text(s,'Validation comparisons, dependent pairs; all pairs retained. No superiority claim.',48,655,1180,30,21,'#657585');
s=slide('Rule demand replaces an assumed human cutoff',
`B counts distinct split tests encountered by one example; L counts all split-node visits, including repeated tests, and M counts prediction trees. The curve shows the fraction of reference examples whose B exceeds each integer budget. A coverage budget b90 is the smallest integer covering at least ninety percent of examples, and b95 covers ninety-five percent. For selected Iris models, both budgets are two, ten and nine across the three seeds. Breast Cancer gives two, two and four. These describe reference-data demand, not human capacity. Original aggregate metrics reconcile exactly for all 78 rows. The five-, seven- and nine-rule scenarios remain secondary sensitivity analyses. Neither seven rules nor a fifty-percent exceedance boundary is validated cognitively. Aggregation effort remains unmodeled; controlled structural behavior is the defensible contribution.`,75);
await chart(s,'rule_demand',40,95,1200,525);
text(s,'b90 / b95: minimum budget covering 90% / 95% of reference examples.',48,625,1180,30,23,'#193347',true);
text(s,'Structural demand, not human capacity · Aggregation effort remains unmodeled.',48,663,1180,25,21,'#657585');
const sources='Sources: results/controlled_v1 tables, summary.json and report.md; protocols/controlled_v1.md. Original scores and summary preserved. Optional nested selector evaluation was not run.';
for(const [i,page] of deck.slides.items.entries()){
 page.speakerNotes.text += '\n\n'+sources;
 const png=await deck.export({slide:page,format:'png',scale:2});
 await fs.writeFile(path.join(out,'previews',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const words=notes.join(' ').split(/\s+/).length;
await fs.writeFile(path.join(out,'speaker_notes.md'),`# Five-minute presentation\n\nTarget timing: 40 + 55 + 55 + 75 + 75 = 300 seconds. Approximately ${words} words including headings; rehearse aloud and adjust pauses to your speaking pace. This is a timing estimate, not a measured rehearsal.\n\n${notes.join('\n\n')}\n\n## Sources and reproduction\n\n${sources}\n\nFigures and displayed statistics are loaded from CSV-derived reports. Rebuild previews with scripts/build_presentation.mjs before repackaging with scripts/build_pdf.py. The PDF contains rendered slide images; vector charts are also available separately.\n`);
try{
 const pdf=await deck.export({format:'pdf'});
 await fs.writeFile(path.join(out,'presentation.pdf'),new Uint8Array(await pdf.arrayBuffer()));
 console.log('Created presentation/presentation.pdf');
}catch(error){
 console.warn(`Direct PDF export unavailable: ${error.message}`);
 console.log('Previews updated. Run python scripts/build_pdf.py.');
}
