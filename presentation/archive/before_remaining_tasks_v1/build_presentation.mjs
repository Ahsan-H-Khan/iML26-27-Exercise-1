// Build the development narrative from metric-development-v2 results only.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const moduleName=process.env.CSC_ARTIFACT_MODULE;
const {Presentation,PresentationFile}=await import(moduleName?pathToFileURL(moduleName).href:'@oai/artifact-tool');
const facts=JSON.parse(await fs.readFile('results/metric_development_v2/presentation_summary.json','utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const out=path.resolve('presentation');
const preview=path.join(out,'previews','metric_development_v2');
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
 text(s,appendix?'Appendix':`${deck.slides.items.length} / 12`,55,686,1160,22,15,muted);
 s.speakerNotes.text=note+'\n\nSource: '+source;
 notes.push(`## ${deck.slides.items.length}. ${title}\n\nTarget: ${seconds} seconds${appendix?' (optional appendix)':''}.\n\n${note}\n\nSource: ${source}`);
 entries.push({title,seconds,appendix});return s;
}
function bottom(s,value){text(s,value,60,620,1160,60,25,ink);}
async function chart(s,name,x=35,y=155,w=1210,h=440){
 s.images.add({blob:new Uint8Array(await fs.readFile(`results/metric_development_v2/figures/${name}.png`)),contentType:'image/png',fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
const pct=x=>(x*100).toFixed(2)+'%';
let s=slide('Developing a complexity metric for tree models',
`The project begins with two separate questions. Does a trained model predict accurately enough to be worth considering, and how much structure does it contain? Accuracy answers the first question. A complexity measure is intended to help with the second. I will walk through path length, its extension to ensembles, and an adjustment for features. I then use the description-length idea to motivate CSC, Cognitive-Structural Complexity. Finally, we evaluate the formula on Iris and Breast Cancer. This is a proposed structural measure. We do not yet have human-subject evidence that a lower score makes a model easier to understand. The empirical evaluation therefore focuses on competitive predictive performance and comparison with existing structural counts.`,55);
text(s,'Cognitive-Structural Complexity (CSC)',60,165,1160,70,42,blue,true);
text(s,'How the formula developed\nWhat it measures on Iris and Breast Cancer',60,295,1160,140,34);
text(s,'First check prediction quality. Then compare complexity.',60,510,1160,70,31);

s=slide('Start with Expected Path Length',
`EPL measures the average number of internal decisions encountered by a reference example in one tree. It is not maximum depth, and it excludes the final leaf. In this teaching example, half the reference examples finish after one decision. A quarter finish after two decisions at each of the other two leaves. The expected path length is therefore one and a half decisions. The weights come from the reference examples reaching each leaf, not from assuming all leaves are equally likely. This answers a useful question about a typical prediction. Its limitation is that one path does not describe all the different rules stored in a model, and a single-tree definition needs to be extended before it describes an ensemble.`,70,`Project definition of empirical EPL; implementation checked against scikit-learn decision_path: ${sklearn}. Teaching example, not experimental data.`);
text(s,'EPL = average number of decisions along one tree’s paths',60,120,1160,60,30,blue);
text(s,'First question',190,215,400,55,30,ink,true);
text(s,'Yes → leaf\n50% of examples\n1 decision',75,320,370,155,28);
text(s,'No → second question\nTwo leaves: 25% each\n2 decisions',485,320,610,155,28);
text(s,'EPL = 0.50 × 1 + 0.25 × 2 + 0.25 × 2 = 1.50',60,530,1160,65,32,blue,true);
bottom(s,'Useful for one prediction path; it does not describe all stored rules.');

s=slide('Extend path length to the ensemble',
`Expected Ensemble Path Complexity extends the same measurement across the actual prediction trees. We report two explicitly named versions. EEPC total is the sum of the tree EPL values; it measures all split visits for an ensemble prediction. EEPC mean divides that sum by the actual number of trees. Continuing the teaching example, one tree has EPL 1.5 and a second has EPL 2.0. Total is 3.5, while the mean is 1.75. This distinction matters: total grows when additional trees repeat work, whereas the mean can hide the number of trees. Neither version tells us how much of that work repeats identical logic. In multiclass boosting, we count every class-specific prediction tree, not just the number of boosting stages.`,70,'Project definitions; total and mean conventions locked in protocols/metric_development_v2.md. Teaching example.');
text(s,'Expected Ensemble Path Complexity (EEPC)',60,125,1160,50,30,blue);
text(s,'Tree 1: EPL = 1.50          Tree 2: EPL = 2.00',60,235,1160,65,32);
text(s,'EEPC total = 1.50 + 2.00 = 3.50',60,345,1160,65,34,blue,true);
text(s,'EEPC mean = 3.50 / 2 trees = 1.75',60,445,1160,65,34,blue,true);
bottom(s,'Total counts all visits. Mean describes the average tree. Neither identifies repeated logic.');

s=slide('Ensemble-adjusted complexity (EA)',
`The next proposed step is ensemble-adjusted complexity, EA. We multiply the path term by a logarithmic feature burden. For this revision, the declared convention uses average path length per tree, and F is the number of distinct input features actually used in internal splits. We use log base two of one plus F; unused dataset columns do not add burden. With average path length 1.75 and two features, the feature term is about 1.585, giving EA about 2.77. The total-path version is saved separately as a sensitivity measure. This adjustment recognizes that tracking several inputs may matter, but it is a design assumption, not a calibrated human cost. Counting features still misses how many different thresholds are used on each feature.`,70,'Project EA definition and locked aggregation conventions. Teaching example.');
text(s,'EA = average path length × log₂(1 + features used)',60,145,1160,75,35,blue,true);
text(s,'Continuing the example: 2 trees using 2 input features',60,270,1160,60,29);
text(s,'EA = 1.75 × log₂(3) = 2.77',60,380,1160,75,39,blue,true);
text(s,'“Features used” counts inputs actually tested by the model.',60,505,1160,55,28);
bottom(s,'One feature can still appear in many different rules.');

s=slide('A feature count can miss different rules',
`Consider two questions about the same feature: whether a measurement is at most two, and whether it is at most five. They use one input feature, but they are different split tests. EA's feature count cannot distinguish that difference by itself. Conversely, if exactly the same rule appears several times, storing or describing it repeatedly need not add distinct logic. The same distinction applies to whole trees, although a tree signature must also account for its predictions and output channel. These observations motivate counting distinct split rules and distinct trees. They do not prove a particular logarithm or weight is optimal. The examples on this slide are explanations of the design, not results from the earlier controlled study.`,65,'Teaching illustration; CSC identities are defined in iml_interpretability/tree_utils.py.');
text(s,'measurement ≤ 2       measurement ≤ 5',60,170,1160,70,36,blue,true);
text(s,'One feature. Two different split tests.',60,280,1160,55,32);
text(s,'The same test repeated elsewhere is still one distinct test.',60,395,1160,80,30);
text(s,'This suggests describing the model’s distinct logic.',60,525,1160,65,32,blue);

s=slide('The Minimum Description Length idea',
`Minimum Description Length, or MDL, is an established model-selection principle. In the introductory two-part formulation, we describe the model and then describe the data using the model. We seek a short combined description. A more complex model can fit better but costs more to describe; a very simple model can leave much of the data unexplained. This is useful motivation for thinking about reusable structure. However, formal MDL requires specified codes. CSC does not encode topology, thresholds, leaf predictions and residual data into a valid bit stream. Our logarithmic counts borrow the description-length idea, but they are not a formal MDL score. We therefore do not claim to benchmark CSC against implemented MDL or to derive CSC mathematically from that principle.`,75,`Peter Grünwald, A Tutorial Introduction to the Minimum Description Length Principle, sections 1.3–1.4: ${mdl}`);
text(s,'A two-part description:',60,145,1160,55,30);
text(s,'Describe the model  +  describe the data using it',60,245,1160,85,35,blue,true);
text(s,'MDL balances model complexity and fit through code length.',60,380,1160,85,30);
text(s,'CSC borrows the motivation. It does not calculate a formal code length.',60,520,1160,90,28);
text(s,'Source: Grünwald, A Tutorial Introduction to the MDL Principle, §§1.3–1.4',60,650,1160,30,19,muted);

const w=facts.worked_example;
s=slide('Combine the ideas in CSC',
`The proposed formula keeps mean path length as its first component. It adds log2 of one plus the number of distinct split tests, S, and log2 of the number of distinct full trees, U. S counts rules across the full fitted model, not only the paths visited in the reference data. Tree identities include outputs and output channels. Using an average in the path term separates per-tree path length from the ensemble term. This is an additive design that replaces EA's multiplicative feature burden; it is not an algebraic derivation. For the real Iris depth-two tree selected in seed 42, the validation mean path is ${w.c_path.toFixed(4)}, there are ${w.n_unique_splits} distinct tests and ${w.n_unique_trees} tree. The resulting CSC is ${w.csc.toFixed(4)}. All three weights are one by choice, not human calibration.`,80);
text(s,'CSC = mean path + log₂(1 + S) + log₂(U)',60,140,1160,75,39,blue,true);
text(s,'S = different split tests in the model\nU = different full trees, including their outputs',60,255,1160,110,30);
text(s,'Real example: Iris · selected depth-two tree · run 1',60,415,1160,50,27,muted);
text(s,`${w.c_path.toFixed(2)} + log₂(${1+w.n_unique_splits}) + log₂(${w.n_unique_trees}) = ${w.csc.toFixed(2)}`,60,490,1160,70,39,blue,true);
bottom(s,'Lower CSC means less measured structure. The weights remain assumptions.');

s=slide('Evaluate on trained models',
`We use two bundled classification datasets: Iris has 150 examples, four features and three classes; Breast Cancer has 569 examples, thirty features and two classes. The same twelve configurations per group cover decision trees, random forests and gradient boosting. Three seeds create shared stratified train, validation and test partitions for every candidate within each group. Training uses approximately half the data, validation one quarter, and test one quarter. No learned preprocessing is needed in this pipeline. A prior dummy classifier is fitted on the training split. A candidate is competitive only if validation balanced accuracy beats the dummy and is within two percentage points of the group best. We freeze the minimum-CSC and maximum-accuracy choices on validation, then evaluate them on test without refitting.`,75);
text(s,'Iris: 150 examples, 4 features, 3 classes\nBreast Cancer: 569 examples, 30 features, 2 classes',60,140,1160,105,29);
text(s,'Decision trees · random forests · gradient boosting\n12 configurations × 2 datasets × 3 runs = 72 candidates',60,290,1160,110,29);
text(s,'Train: 50%     Validation: 25%     Test: 25%',60,440,1160,60,32,blue,true);
text(s,'Competitive = beats the dummy baseline and is within\n2 percentage points of the best validation balanced accuracy.',60,545,1160,100,27);

s=slide('First, check predictive performance',
`Each point is one validation candidate. Balanced accuracy means the average recall across classes, so predicting only the majority class produces one-third for Iris and one-half for Breast Cancer. All ${facts.above_dummy_count} candidates beat those simple baselines. That alone is a low bar: the Iris one-split trees reach only about two-thirds. We therefore also apply the within-two-percentage-point rule separately in each dataset and seed. ${facts.eligible_count} of ${facts.candidate_count} candidates pass both conditions: fifteen for Iris and twenty for Breast Cancer. They are competitive for this benchmark, not certified adequate for a real-world application. The spread includes different settings and random splits; these points should not be treated as independent repeated measurements of one model.`,70);
text(s,`${facts.above_dummy_count}/72 beat the dummy baseline; ${facts.eligible_count}/72 also meet the near-best rule.`,60,115,1160,55,28);
await chart(s,'predictive_performance',35,175,1210,420);
bottom(s,'Balanced accuracy = average recall across classes. All points are validation results.');

s=slide('Then compare accuracy and CSC',
`Now we apply CSC to the candidates. Faint points are outside the competitive set, solid points are inside it, and black outlines identify the minimum-CSC choice for each seed. Both panels use the same scales. The useful region is high predictive accuracy with lower measured complexity, but the eligibility threshold is set within each seed rather than as one line across the whole panel. On Iris, the accuracy-only and CSC selectors choose the same models in all three runs. On Breast Cancer, CSC chooses single trees instead of the accuracy-only ensemble choices. The validation accuracy costs are ${facts.groups.filter(g=>g.dataset==='breast_cancer').map(g=>g.validation_ba_cost_pp.toFixed(2)).join(', ')} percentage points. This is the declared trade-off, not evidence that lower complexity improves predictions or human understanding.`,75);
text(s,'Solid points: competitive models. Outlined points: minimum-CSC choices.',60,110,1160,55,26);
await chart(s,'accuracy_csc',35,170,1210,425);
bottom(s,'Iris: both selectors agree. Breast Cancer: CSC selects smaller single trees.');

s=slide('Does CSC rank models differently?',
`We compare rankings only among competitive models, separately within each dataset and seed. The table shows Spearman rank correlation with CSC. One means exactly the same order. CSC agrees closely with node count in all six groups; it also generally agrees with total ensemble path length. The EA ranking differs more in the groups with several eligible models. This is informative about which structural dimensions the formulas emphasize, but it is not evidence that CSC's order is the correct one for people. Iris runs two and three each contain only two eligible models, so perfect rank correlations there are especially weak evidence. The report also retains mean path length and EA based on total visits as sensitivities. We make no significance claim from the overlapping splits.`,75);
await chart(s,'metric_comparison',45,145,1190,425);
text(s,'CSC mostly agrees with node count; EA gives more different rankings.',60,585,1160,50,27);
text(s,'Iris runs 2 and 3 contain only two eligible models each.',60,645,1160,35,22,muted);

s=slide('What do the test results show?',
`These test results concern only choices frozen using validation data. On Iris, both selectors chose the same models and therefore have identical results. On Breast Cancer, the lower-CSC choices have a lower mean complexity but also lower mean test balanced accuracy than the accuracy-only choices. This is a real trade-off in this evaluation, not a free improvement. The displayed results are means across three overlapping splits; detailed sample standard deviations and individual rows are in the report. A two-point tolerance on validation does not guarantee that tolerance on each test split. The data were already inspected in prior project work, so this is descriptive follow-up rather than a new confirmatory experiment. We have developed a reproducible structural score and evaluated it on competitive models, but have not validated human interpretability or superiority over simpler metrics.`,80);
text(s,'Mean results for choices frozen on validation data',60,115,1160,50,28);
const testRows=facts.test_summary;
text(s,'Dataset / selection',60,220,485,45,25,muted,true);
text(s,'Test balanced accuracy',580,220,365,45,25,muted,true);
text(s,'Test CSC',1000,220,215,45,25,muted,true);
let iy=280;
for(const dataset of ['iris','breast_cancer']){
 for(const selector of ['accuracy_only','minimum_csc']){
  const r=testRows.find(r=>r.dataset===dataset&&r.selector===selector);
  text(s,`${dataset==='iris'?'Iris':'Breast Cancer'} · ${selector==='accuracy_only'?'accuracy choice':'CSC choice'}`,60,iy,490,45,26);
  text(s,pct(r.mean_test_ba),580,iy,360,45,29);
  text(s,r.mean_test_csc.toFixed(2),1000,iy,210,45,29);
  iy+=65;
 }
}
text(s,'Lower complexity can cost accuracy. CSC still needs human validation.',60,580,1160,55,27);
text(s,'Three overlapping splits; descriptive means, not an independent confirmation.',60,650,1160,30,21,muted);

s=slide('Appendix: components of the selected models',
`These bars use the six minimum-CSC validation choices. Each bar adds mean path length, log2 of one plus distinct rules, and log2 of distinct full trees. Runs one, two and three correspond to seeds 42, 1337 and 2026. Iris selects a depth-two decision tree, then two five-tree depth-two forests. Breast Cancer selects depth-two trees twice and a depth-four tree once. Scores depend on the reference paths and stored representation. The figure is a decomposition, not evidence that the components require equal human effort.`,45,dataSource,true);
await chart(s,'components',45,150,1190,445);
bottom(s,'Rule term = log₂(1 + S). Tree term = log₂(U).');

s=slide('Appendix: definitions and sources',
`The presentation distinguishes established motivation from project choices. Grünwald supports the two-part MDL explanation; scikit-learn documentation supports the path-extraction implementation check. EEPC aggregation, EA feature burden and the additive CSC combination are explicit project conventions. The full protocol, reports and saved tables provide reproducibility. No historical controlled-study outputs feed this deck.`,30,'References listed on this slide and in docs/metric_development_sources.md.',true);
text(s,'MDL motivation',60,145,1160,45,30,blue,true);
text(s,'Peter Grünwald · A Tutorial Introduction to the MDL Principle\nSections 1.3–1.4: model code + data-given-model code',60,210,1160,95,27);
text(s,mdl,60,315,1160,35,21,muted);
text(s,'Path implementation',60,390,1160,45,30,blue,true);
text(s,'scikit-learn · DecisionTreeClassifier.decision_path\nMetric definitions and results: protocols/metric_development_v2.md',60,455,1160,100,26);
text(s,'EEPC, EA and CSC are project definitions. No formal MDL baseline was computed.',60,600,1160,65,24);

for(const [i,page] of deck.slides.items.entries()){
 const filename=`slide-${String(i+1).padStart(2,'0')}.png`;
 const png=await deck.export({slide:page,format:'png',scale:2});
 await fs.writeFile(path.join(preview,filename),new Uint8Array(await png.arrayBuffer()));
 entries[i].preview=`previews/metric_development_v2/${filename}`;
 console.log(`Rendered ${i+1}/${entries.length}: ${entries[i].title}`);
}
await fs.writeFile(path.join(out,'slide_manifest.json'),JSON.stringify({version:'metric_development_v2',width:1280,height:720,slides:entries},null,2));
const duration=entries.filter(e=>!e.appendix).reduce((a,e)=>a+e.seconds,0);
await fs.writeFile(path.join(out,'speaker_notes.md'),`# CSC: development and evaluation\n\n12 main slides plus 2 optional appendix slides. Main target: ${duration} seconds (${(duration/60).toFixed(1)} minutes); rehearse to set actual timing. This longer working deck follows the user's instruction; the original course's five-minute submission requirement has not been changed.\n\n${notes.join('\n\n')}\n`);
await fs.mkdir(path.join(out,'.build'),{recursive:true});
try{
 const draft=await PresentationFile.exportPptx(deck);
 await draft.save(path.join(out,'.build','metric_development_v2_draft.pptx'));
 console.log('Private editable draft exported.');
}catch(e){console.warn('Private draft export unavailable: '+e.message);}
console.log('Run scripts/build_pdf.py to package the manifest-listed slides.');
