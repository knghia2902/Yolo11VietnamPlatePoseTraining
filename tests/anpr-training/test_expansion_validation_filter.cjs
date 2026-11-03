const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_expansion.html'),'utf8');
const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
new vm.Script(script);
function element(){return {value:'',options:[],append(x){this.options.push(x)},before(){},setAttribute(){}}}
const nodes=new Map();function node(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)}
let loads=0;
const context=vm.createContext({$:node,document:{createElement:element},load(){loads++}});
vm.runInContext("let reviewSection='point',offset=15,resumeAnchor='old';",context);
const start=script.indexOf('const pointStatuses=');const end=script.indexOf('function applySection(',start);
vm.runInContext(script.slice(start,end),context);
assert(node('status').options.some(o=>o.value==='validation'&&o.textContent==='Validation'));
assert(vm.runInContext("pointStatuses.has('validation')",context));
node('kind').value='validation';node('status').value='corners_done';node('kind').onchange();
assert.equal(node('status').value,'validation');assert.equal(vm.runInContext('offset',context),0);
node('status').value='corners';node('status').onchange();assert.equal(node('kind').value,'all');
node('kind').value='acs';node('status').value='validation';node('status').onchange();assert.equal(node('kind').value,'validation');
assert.equal(loads,3);
console.log('Validation filter: HTML script syntax, option visibility setup, source/status synchronization and cursor reset pass');
