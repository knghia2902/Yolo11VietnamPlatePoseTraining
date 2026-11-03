const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_expansion.html'),'utf8');
const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
const code=script.slice(script.indexOf('async function approveCorners()'),script.indexOf('function remainsInFilter('));
const nodes=new Map();function node(id){if(!nodes.has(id))nodes.set(id,{value:'',checked:false});return nodes.get(id)}
const writes=[];const stored=new Map();
const context=vm.createContext({$:node,structuredClone,names:{},message(){},frozenCorners(){return false},
 localStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)},
 api:async(url,body)=>{writes.push(body);return {id:body.id,version:body.version+1,kind:body.action==='approve_validation'?'validation':'acs',status:'corners_done'}},
 setBusy(v){vm.runInContext(`busy=${v}`,context)}});
vm.runInContext("let current={id:'fixture',version:0},busy=false,dirty=true,points=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]];",context);
vm.runInContext(code,context);node('reviewer').value='Tester';node('single').checked=true;
(async()=>{
 node('approvalTarget').value='validation';assert.equal(await vm.runInContext('approveCorners()',context),true);
 assert.equal(writes[0].action,'approve_validation');assert.equal(writes[0].edit.single_plate_reviewed,true);
 node('approvalTarget').value='train';assert.equal(await vm.runInContext('approveCorners()',context),true);
 assert.equal(writes[1].action,'approve_corners');
 assert.equal(JSON.parse(stored.get('cornerApprovalUndoStack')).length,2);
 assert.equal(vm.runInContext('busy',context),false);
 console.log('Approval target: Train/Validation request routing, point payload and undo stack pass');
})().catch(e=>{console.error(e);process.exitCode=1});
