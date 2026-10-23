const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_pose.html'),'utf8');
const row=id=>({id,image:`images/${id}.png`,aliases:[id],points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],reviewer:'tester',group:'source',reviewed:false,group_reviewed:false,exclude:false});
const nodes=new Map(),ctx=new Proxy({},{get:()=>()=>{}});
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',getContext:()=>ctx,getBoundingClientRect:()=>({left:0,top:0,width:100,height:100}),click(){}});return nodes.get(id)}
const sandbox={document:{getElementById:node,createElement:()=>({click(){}})},Image:class{naturalWidth=100;naturalHeight=100;set src(_value){this.onload()}},setTimeout:()=>1,clearTimeout:()=>{},alert:()=>{},URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{}};
vm.createContext(sandbox);vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1].replace('/*MANIFEST*/null',JSON.stringify({records:[row('a'),row('b')]})),sandbox);
function key(key,extra={}){let prevented=false;const event={key,target:{tagName:'BODY'},preventDefault(){prevented=true},...extra};sandbox.document.onkeydown(event);return prevented}
assert.equal(key('ArrowRight'),true);assert.equal(vm.runInContext('index',sandbox),1);
assert.equal(key('ArrowLeft'),true);assert.equal(vm.runInContext('index',sandbox),0);
for(const tagName of ['INPUT','TEXTAREA','SELECT'])assert.equal(key('Enter',{target:{tagName}}),false);
assert.equal(key('ArrowRight',{target:{isContentEditable:true}}),false);
for(const flag of ['ctrlKey','altKey','metaKey','shiftKey','isComposing'])assert.equal(key('Enter',{[flag]:true}),false);
assert.equal(key('Enter',{repeat:true}),true);assert.equal(vm.runInContext('rows[0].reviewed',sandbox),false);
assert.equal(key('Enter'),true);assert.equal(vm.runInContext('rows[0].reviewed',sandbox),true);assert.equal(vm.runInContext('index',sandbox),1);
vm.runInContext('actionBusy=true',sandbox);key('ArrowLeft');assert.equal(vm.runInContext('index',sandbox),1);
vm.runInContext('actionBusy=false;unsaved=1',sandbox);key('ArrowLeft');assert.equal(vm.runInContext('index',sandbox),1);
vm.runInContext('unsaved=0;drag={}',sandbox);key('Enter');assert.equal(vm.runInContext('rows[1].reviewed',sandbox),false);
console.log('Keyboard: Enter approval, arrows, input/IME/modifier guards, repeat and pending-save guards pass');
