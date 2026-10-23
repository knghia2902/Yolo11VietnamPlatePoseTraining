// Logic tests with a synthetic DOM stub. No browser, customer data or network.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_pose.html'), 'utf8');
const row = id => ({id, image:`images/${id}.png`, aliases:[`${id}.png`], points:[], group:'source', reviewed:false, group_reviewed:false, exclude:false, proposal:'unreviewed'});
const manifest = {source_sha256:'test', records:[row('a'),row('b')]};
const nodes = new Map();
const ctx = new Proxy({}, {get:(_,key)=>()=>{}});
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {value:'', checked:false, textContent:'', files:[],
    getContext:()=>ctx, getBoundingClientRect:()=>({left:0,top:0,width:100,height:100}),click(){}});
  return nodes.get(id);
}
const alerts=[];
const sandbox = {document:{getElementById:node, createElement:()=>({click(){}})},
  Image:class {naturalWidth=100;naturalHeight=100;set src(value){this.onload()}},
  alert:text=>alerts.push(text), URL:{createObjectURL:()=>'',revokeObjectURL(){}},
  Blob:class {},setTimeout:fn=>fn()};
vm.createContext(sandbox);
const script = source.match(/<script>([\s\S]*)<\/script>/)[1].replace('/*MANIFEST*/null', JSON.stringify(manifest));
vm.runInContext(script,sandbox);
node('approve').onclick();
assert.equal(alerts.length,1,'Incomplete corners must not be approved');
for (const [x,y] of [[20,20],[80,20],[80,80],[20,80]]) node('canvas').onclick({clientX:x,clientY:y});
node('reviewer').value='tester';
node('approve').onclick();
assert.equal(vm.runInContext('rows[0].reviewed',sandbox),true);
assert.equal(vm.runInContext('rows[0].group_reviewed',sandbox),true);
assert.equal(vm.runInContext('index',sandbox),1,'Approval advances to next image');
node('prev').onclick();
node('group').value='new-session';
node('next').onclick();
assert.equal(vm.runInContext('rows[0].reviewed',sandbox),false,'Changing group invalidates approval');
node('prev').onclick();
assert.equal(node('group').value,'new-session');
node('clear').onclick();
assert.equal(vm.runInContext('rows[0].points.length',sandbox),0);
node('export').onclick();
(async () => {
  const saved = JSON.parse(vm.runInContext('JSON.stringify(rows)', sandbox));
  saved[0].points = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]];
  saved[0].reviewer = 'restored tester';
  node('load').files = [{text:async()=>JSON.stringify({source_sha256:'test',records:saved})}];
  await node('load').onchange({target:node('load')});
  assert.equal(vm.runInContext('rows[0].points.length',sandbox),4);
  assert.equal(node('reviewer').value,'restored tester');
  node('load').files = [{text:async()=>JSON.stringify({source_sha256:'wrong',records:saved})}];
  await node('load').onchange({target:node('load')});
  assert.equal(alerts.length,2,'Wrong-source reviews rejected');
  console.log('Review logic: editing, approval, invalidation, navigation, reset, export, reload and wrong-source rejection pass');
})().catch(error=>{console.error(error);process.exitCode=1});
