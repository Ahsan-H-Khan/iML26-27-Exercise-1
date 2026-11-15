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
 text(s,`${deck.slides.items.length} / 5`,48,690,1180,22,14,'#657585');
 s.speakerNotes.text=note;
 notes.push(`## Slide ${deck.slides.items.length} — ${title} (${seconds} seconds)\n\n${note}`);
 return s;
}
async function chart(s,name,x,y,w,h){
 s.images.add({blob:new Uint8Array(await fs.readFile(`results/controlled_v1/figures/plain_slides/${name}.png`)),contentType:'image/png',
              fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
const example=JSON.parse(await fs.readFile('results/controlled_v1/figures/plain_slides/example.json','utf8'));
let s=slide('How much logic is inside a model?',
`A decision tree makes a prediction by asking a sequence of questions about its input. CSC, short for Cognitive-Structural Complexity, is our proposed score for how much logic a tree model contains. It adds the average number of questions encountered per tree and example, log2 of one plus the number of distinct split tests, and log2 of the number of distinct full trees. Those weights are assumptions. A smaller score means less measured structural complexity, not proven ease of understanding. We test its behavior using constructed examples and compare models trained on Iris and Breast Cancer.`,40);
text(s,'A tree model predicts by asking questions.',60,145,1160,65,34);
text(s,'For example: “Is this measurement above a threshold?”',60,225,1160,65,28,'#657585');
text(s,'CSC combines three things:',60,340,1160,48,30);
text(s,'Questions along a path\nDifferent rules\nDifferent trees',90,415,1070,165,32,'#0072B2');
text(s,'Lower score = less measured complexity. Human understanding is untested.',60,635,1160,45,23);
s=slide('The same model size can hide different paths',
`Here we hold predictions and size fixed: both constructed trees have seven nodes, four leaves and the same classification function. One tree is balanced, so every prediction takes two questions. The chain-shaped tree can finish after one question or require three. We change which examples arrive. When most exit early, chain CSC is 3.30; when most go deeper, it rises to 4.85. The balanced tree stays at four. The uniform-reference case, 4.25 for the chain, is in the full report. Counting nodes misses this difference, although average path length also captures it. Lower CSC is the direction of lower measured complexity.`,55);
text(s,'Two trees. Same predictions. Seven nodes each.',60,110,1160,50,29);
await chart(s,'paths',70,185,1140,420);
text(s,'The score depends on the paths the examples actually take.',60,635,1160,45,27);
s=slide('Repeating a rule adds no new rule',
`We keep twelve one-question trees, called stumps, and thirty-six total nodes. Every example still visits twelve split nodes. Only the number of distinct rules changes. When all twelve trees repeat one rule, CSC is two. When all twelve use different rules, CSC is 8.29. This controlled setup shows what simple node counts miss. These are artificial averaging ensembles, and their predictions can change as rule diversity changes. A separate control repeats each of three base trees equally: its probabilities and CSC stay fixed while nodes and visits grow. That invariance applies to averaging, not arbitrary boosting duplication.`,55);
text(s,'12 one-question trees in every example below.',60,110,1160,50,29);
await chart(s,'diversity',105,185,1070,420);
text(s,'Same tree count and node count. More different rules, higher CSC.',60,635,1160,45,26);
s=slide('The measures sometimes disagree',
`This is an illustrative disagreement selected after looking at the saved comparisons, not a representative sample. Both models use the same Breast Cancer validation split, seed 42. Their balanced accuracy is identical at ${(example.balanced_accuracy_a*100).toFixed(1)} percent and both are within two percentage points of the best candidate. Gradient boosting uses fifty depth-one trees; random forest uses five trees with maximum depth five. Node count favors the forest, while CSC slightly favors boosting. Total expected path length also favors the forest. CSC weights average paths and distinct logic differently from raw size. Across all 116 near-best pairs, CSC reverses node-count rankings only three times and TEPL rankings seven times. These pairs overlap and are not independent experiments. We have not established which ranking better matches human understanding.`,75);
text(s,'One example: same validation accuracy, different complexity rankings.',60,110,1160,55,27);
text(s,'Breast Cancer · same data split · selected to illustrate a disagreement',60,175,1160,35,21,'#657585');
text(s,'Gradient boosting',545,265,295,45,27,'#0072B2',true);
text(s,'Random forest',930,265,290,45,27,'#0072B2',true);
const rows=[['Balanced accuracy¹',`${(example.balanced_accuracy_a*100).toFixed(1)}%`,`${(example.balanced_accuracy_b*100).toFixed(1)}%`],['Total nodes',`${example.total_nodes_a}`,`${example.total_nodes_b}  ← fewer`],['CSC score',`${example.csc_a.toFixed(2)}  ← lower`,`${example.csc_b.toFixed(2)}`]];
rows.forEach(([label,a,b],i)=>{const y=345+i*75;text(s,label,60,y,440,48,29);text(s,a,545,y,300,48,30);text(s,b,930,y,290,48,30);});
text(s,'CSC and node count reverse order in just 3 of 116 near-best pairs.',60,605,1160,42,27);
text(s,'¹ Accuracy averaged across classes. Lower nodes / CSC means less measured complexity.',60,660,1160,30,20,'#657585');
s=slide('How many different tests does a prediction need?',
`Each bar shows the smallest number of distinct split tests sufficient for at least ninety percent of validation examples, for one frozen selected model. Run one, two and three correspond to seeds 42, 1337 and 2026. For Breast Cancer the selected models need two, two and four tests. For Iris they need two, ten and nine. Runs can select different models, which explains part of that variation: Iris selects a single tree first, then two five-tree forests. These are measurements of model demand, not limits on human memory. The ninety-five-percent values happen to be the same in these six cases. Full exceedance curves, all visits and tree counts are retained in the report. Output aggregation effort is still omitted. We can describe structural differences; we cannot yet say how hard people will find these models.`,75);
text(s,'Tests needed to cover 90% of examples in each selected model.',60,110,1160,50,28);
await chart(s,'demand',65,180,1150,425);
text(s,'These are model measurements, not limits on human memory.',60,635,1160,45,27);
const sources='Sources: unchanged results/controlled_v1 CSV tables and summary.json. Q3 uses the saved Breast Cancer seed-42 gb_n50_d1 / rf_n5_d5 pair, deliberately chosen to explain a disagreement. All pairs remain in the report. Run labels 1/2/3 map to seeds 42/1337/2026. Optional nested evaluation was not run.';
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
