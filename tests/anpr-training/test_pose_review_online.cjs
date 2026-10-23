// Synthetic DOM and fetch tests; no real browser or customer network/data.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_pose.html'),'utf8');
const row=id=>({id,image:`images/${id}.png`,aliases:[id],points:[],group:'source',reviewed:false,group_reviewed:false,exclude:false,proposal:'test',_version:0});
const manifest={source_sha256:'test',records:[row('a'),row('b')]};
const nodes=new Map(),canvasContext=new Proxy({},{get:()=>()=>{}}),requests=[];
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',checked:false,getContext:()=>canvasContext,getBoundingClientRect:()=>({left:0,top:0,width:100,height:100}),click(){}});return nodes.get(id)}
let fail=false,unload;
const sandbox={document:{getElementById:node,createElement:()=>({click(){}})},
 Image:class{naturalWidth=100;naturalHeight=100;set src(_value){this.onload()}},
 window:{addEventListener:(_name,handler)=>{unload=handler}},
 setTimeout:()=>1,clearTimeout:()=>{},alert:()=>{},
 fetch:async(_url,options)=>{const payload=JSON.parse(options.body);requests.push(payload);return {ok:!fail,status:fail?409:200,json:async()=>({version:payload.version+1})}},
 URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{}};
vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1].replace('/*MANIFEST*/null',JSON.stringify(manifest)).replace('/*SERVER*/null','{"cursor":0}'),sandbox);
(async()=>{
 for(const [x,y]of [[20,20],[80,20],[80,80],[20,80]])node('canvas').onclick({clientX:x,clientY:y});
 node('reviewer').value='tester';
 await node('save').onclick();
 assert.equal(requests.length,1);assert.equal(requests[0].edit.points.length,4);
 assert.equal(vm.runInContext('rows[0]._version',sandbox),1);
 const approval=node('approve').onclick();node('approve').onclick();await approval;
 assert.equal(requests.at(-1).edit.reviewed,true);
 assert.equal(requests.at(-1).edit.group_reviewed,true,'Online approval automatically confirms and persists group');
 assert.equal(requests.length,2,'Double click cannot approve two images');
 assert.equal(requests.at(-1).id,'a','Approval saves the original image');
 assert.equal(requests.at(-1).cursor,1,'Resume position is next image');
 assert.equal(vm.runInContext('index',sandbox),1,'Approval advances after save');
 vm.runInContext('queueSave();queueSave()',sandbox);await vm.runInContext('pending',sandbox);
 assert.deepEqual(requests.map(r=>r.version),[0,1,0,1]);
 await node('exclude').onclick();
 assert.equal(requests.at(-1).edit.exclude,true,'Exclusion saves immediately');
 assert.equal(vm.runInContext('index',sandbox),0,'Exclusion moves to next image after save');
 node('prev').onclick();await vm.runInContext('pending',sandbox);
 assert.equal(vm.runInContext('index',sandbox),0,'Navigation skips excluded image');
 fail=true;node('next').onclick();await vm.runInContext('pending.catch(()=>{})',sandbox);
 assert.equal(vm.runInContext('index',sandbox),0,'Conflict prevents navigation and lost edits');
 assert.match(node('save-status').textContent,/CHƯA LƯU/);
 let prevented=false;unload({preventDefault(){prevented=true}});assert.equal(prevented,true);
 node('export').onclick();
 console.log('Online save: snapshot, approval, ordered revisions, conflict block, unload guard and backup export pass');
})().catch(error=>{console.error(error);process.exitCode=1});
