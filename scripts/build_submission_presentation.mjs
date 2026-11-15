// Build the development narrative from metric-development-v2 results only.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const moduleName=process.env.CSC_ARTIFACT_MODULE;
const {Presentation,PresentationFile}=await import(moduleName?pathToFileURL(moduleName).href:'@oai/artifact-tool');
const facts=JSON.parse(await fs.readFile('results/metric_development_v2/presentation_summary.json','utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const out=path.resolve('presentation');
const preview=path.join(out,'previews','completion_short');
await fs.mkdir(preview,{recursive:true});
const ink='#26384a',blue='#0072B2',muted='#596b7b';
const notes=[], entries=[];
const mdl='https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf';
const sklearn='https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html';
const dataSource='results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.';
function text(s,value,x,y,w,h,size=28,color=ink,bold=false){
 const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 box.text=value;box.text.style={fontFamily:'Arial',fontSize:size,color,bold};return box;
}
function slide(title,note,seconds=65,source=dataSource,appendix=false){
 const s=deck.slides.add();s.background.fill='#ffffff';
 text(s,title,52,30,1176,78,36,ink,true);
 text(s,appendix?'Appendix':`${deck.slides.items.length} / 7`,55,686,1160,22,15,muted);
 s.speakerNotes.text=note+'\n\nSource: '+source;
 notes.push(`## ${deck.slides.items.length}. ${title}\n\nTarget: ${seconds} seconds${appendix?' (optional appendix)':''}.\n\n${note}\n\nSource: ${source}`);
 entries.push({title,seconds,appendix});return s;
}
function bottom(s,value){text(s,value,60,620,1160,60,25,ink);}
async function chart(s,name,x=35,y=155,w=1210,h=440){
 s.images.add({blob:new Uint8Array(await fs.readFile(`results/metric_development_v2/figures/${name}.png`)),contentType:'image/png',fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
const pct=x=>(x*100).toFixed(2)+'%';

let s=slide('CSC: complexity after checking accuracy',
`The task is to propose an interpretability metric and evaluate it. My measure summarizes tree-model structure. EPL counts average path decisions. EEPC extends paths over an ensemble. EA multiplies the mean path by a feature burden. The description-length idea motivates counting distinct rules and trees. CSC is the sum shown here; it is MDL-inspired rather than a formal code length or calibrated human score.`,40);
text(s,'CSC = mean path + log₂(1 + S) + log₂(U)',60,160,1160,80,38,blue,true);
text(s,'S: distinct split tests in the model\nU: distinct full trees, including their outputs',60,285,1160,125,31);
text(s,'First retain competitive predictions. Then compare CSC.',60,480,1160,80,32);
bottom(s,'Lower scores mean less measured structure. Human effort is uncalibrated.');

s=slide('Prediction quality and the complexity trade-off',
`We use Iris and Breast Cancer, with decision trees, random forests and gradient boosting. The original experiment has 72 validation candidates over three splits. All beat a prior-class dummy; 35 meet the two-percentage-point tolerance. Minimum CSC is selected within that competitive set. The table shows means on test sets for validation-frozen choices. On Iris both selectors agree. On Breast Cancer, lower complexity comes with lower test accuracy. Three overlapping splits make these descriptive means.`,55);
text(s,'72 candidates beat the dummy; 35 also meet the near-best rule.',60,135,1160,60,28);
text(s,'Dataset / choice',60,240,510,45,26,muted,true);
text(s,'Test balanced accuracy',590,240,360,45,25,muted,true);
text(s,'CSC',1040,240,160,45,26,muted,true);
let yy=305;
for(const [dataset, selector] of [['iris','minimum_csc'],['breast_cancer','accuracy_only'],['breast_cancer','minimum_csc']]){
 const r=facts.test_summary.find(r=>r.dataset===dataset&&r.selector===selector);
 text(s,dataset==='iris'?'Iris / both choices':`Breast Cancer / ${selector==='accuracy_only'?'accuracy':'CSC'}`,60,yy,520,50,28);
 text(s,pct(r.mean_test_ba),590,yy,350,50,30);
 text(s,r.mean_test_csc.toFixed(2),1040,yy,150,50,30);yy+=75;
}
bottom(s,'Near-best validation accuracy does not guarantee equal test accuracy.');

s=slide('The threshold investigation',
`The hypothetical budget concerns tests encountered by one prediction, not the CSC score. We flag models if more than half their validation predictions need more than seven distinct tests. In both datasets 21 of 36 candidates are flagged. CSC ranges overlap across that boundary, so one increasing CSC cutoff cannot classify all models correctly. Five- and nine-test sensitivity analyses also give different flags. These assumed budgets are not established human capacities.`,40,'results/threshold_v1/pooled_thresholds.csv');
text(s,'Hypothesis: over half of predictions need more than 7 tests',60,150,1160,75,30,blue,true);
text(s,'21 / 36 models flagged in Iris\n21 / 36 models flagged in Breast Cancer',60,285,1160,130,34);
text(s,'CSC scores overlap across this boundary in both datasets.',60,485,1160,85,32);
bottom(s,'The study does not establish a universal interpretability cutoff.');

s=slide('Training examples and model complexity',
`Task 2 changes training samples, features and class balance while holding validation examples fixed. We completed 756 fits across 63 unique conditions, retaining all three seeds and both datasets. This slide compares quarter-sized training samples with the original training set while holding each configuration fixed. Average CSC falls in both datasets, with small average accuracy losses. These averages summarize 36 configuration-and-seed comparisons per dataset; they are not independent replicates. Choosing different competitive models can reverse the CSC trend.`,45,'results/metafeatures_v1/report.md');
text(s,'25% of training examples versus the original training set',60,135,1160,65,29);
text(s,'Same model configurations',60,250,550,50,29,blue,true);
text(s,'Iris: CSC −0.63 on average; accuracy −1.00 pp\nBreast Cancer: CSC −1.93; accuracy −0.66 pp',60,345,1160,125,30);
bottom(s,'The selected model can switch, so its CSC need not follow the same trend.');

s=slide('Feature removal can lose predictive information',
`This Iris plot compares nested feature subsets, from one feature to all four. Solid lines show the chosen model; dashed lines show the best validation accuracy available. One-feature accuracy varies greatly across seeds because the retained feature differs. Relative to the original choice, selected accuracy changes are minus46.15, minus23.29, and zero percentage points. The Breast Cancer effects are smaller, but CSC still does not change monotonically. Feature availability changes information as well as feature count.`,45,'results/metafeatures_v1/selected_endpoint_comparisons.csv');
s.images.add({blob:new Uint8Array(await fs.readFile('results/metafeatures_v1/figures/slide_features_iris.png')),contentType:'image/png',fit:'contain',position:{left:35,top:150,width:1210,height:430}});
bottom(s,'Iris: which feature remains matters. A lower CSC alone is insufficient.');

s=slide('Class imbalance needs a size-matched comparison',
`We retain only a quarter of the designated training class and compare with a stratified training subset of exactly the same total size. The results hold model configuration fixed and average the 36 configuration-and-seed pairs per dataset. Accuracy falls by about five points on Iris and3.6 on Breast Cancer. Mean CSC barely changes on Iris and decreases on Breast Cancer. Reduced complexity therefore does not imply improved prediction or easier human understanding.`,45,'results/metafeatures_v1/imbalance_paired_comparisons.csv');
text(s,'25% class retention versus a control with the same sample count',60,145,1160,80,29,blue,true);
text(s,'Iris: accuracy −4.99 pp; CSC −0.04\nBreast Cancer: accuracy −3.59 pp; CSC −0.93',60,300,1160,145,32);
bottom(s,'These are mean fixed-configuration differences across three splits.');

s=slide('What the project establishes',
`CSC gives a reproducible structural comparison after checking predictive quality. The proposed budget does not establish a universal cutoff. Dataset metadata affects learned models, but direction depends on the data, seed and whether the model configuration is held fixed. Twelve of63 conditions lose the original accuracy reference. Human interpretability and generalization beyond these two datasets remain open. The long deck and reports contain the detailed definitions, plots and limitations.`,30,'results/metafeatures_v1/report.md; results/threshold_v1/report.md');
text(s,'CSC supports structural comparison among competitive models.',60,155,1160,95,34,blue,true);
text(s,'Dataset changes can lower complexity and accuracy together.\n12 of 63 conditions lose the original accuracy reference.',60,320,1160,140,30);
bottom(s,'Human interpretability still requires direct evaluation.');
for(const [i,page] of deck.slides.items.entries()){
 const filename=`slide-${String(i+1).padStart(2,'0')}.png`;
 const png=await deck.export({slide:page,format:'png',scale:2});
 await fs.writeFile(path.join(preview,filename),new Uint8Array(await png.arrayBuffer()));
 entries[i].preview=`previews/completion_short/${filename}`;
 console.log(`Rendered ${i+1}/${entries.length}: ${entries[i].title}`);
}
await fs.writeFile(path.join(out,'slide_manifest.json'),JSON.stringify({version:'completion_short',width:1280,height:720,slides:entries},null,2));
const duration=entries.filter(e=>!e.appendix).reduce((a,e)=>a+e.seconds,0);
await fs.writeFile(path.join(out,'speaker_notes.md'),`# CSC: development and evaluation\n\n7 slides. Main target: ${duration} seconds (${(duration/60).toFixed(1)} minutes); rehearse to set actual timing. Five-minute submission version; target timing is 300 seconds and requires rehearsal.\n\n${notes.join('\n\n')}\n`);
await fs.mkdir(path.join(out,'.build'),{recursive:true});
try{
 const draft=await PresentationFile.exportPptx(deck);
 await draft.save(path.join(out,'.build','completion_short_draft.pptx'));
 console.log('Private editable draft exported.');
}catch(e){console.warn('Private draft export unavailable: '+e.message);}
console.log('Run scripts/build_pdf.py to package the manifest-listed slides.');
