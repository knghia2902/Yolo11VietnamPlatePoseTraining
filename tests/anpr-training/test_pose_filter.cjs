// Filter and save-order tests with synthetic annotations and fetch responses.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_pose.html'),'utf8');
function create(online){
 const row=(id,reviewed=false,exclude=false)=>({id,image:`images/${id}.png`,aliases:[id],points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],reviewer:'tester',group:'source',reviewed,group_reviewed:true,exclude,_version:0});
 const records=[row('a'),row('b',true),row('c',false,true),row('d')];
 const nodes=new Map(),ctx=new Proxy({},{get:()=>()=>{}}),requests=[];
 function node(id){if(!nodes.has(id))nodes.set(id,{value:'',getContext:()=>ctx,getBoundingClientRect:()=>({left:0,top:0,width:100,height:100}),click(){}});return nodes.get(id)}
 const state={fail:false};
 const sandbox={document:{getElementById:node,createElement:()=>({click(){}})},Image:class{naturalWidth=100;naturalHeight=100;set src(_value){this.onload()}},window:{addEventListener(){}},setTimeout:()=>1,clearTimeout:()=>{},alert:()=>{},URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{},fetch:async(_url,options)=>{const payload=JSON.parse(options.body);requests.push(payload);return{ok:!state.fail,status:state.fail?409:200,json:async()=>({version:payload.version+1})}}};
 vm.createContext(sandbox);vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1].replace('/*MANIFEST*/null',JSON.stringify({records})).replace('/*SERVER*/null',online?'{}':'null'),sandbox);
 return{node,requests,state,get:code=>vm.runInContext(code,sandbox),filter:async value=>{node('review-filter').value=value;await node('review-filter').onchange()}};
}
(async()=>{
 for(const online of [false,true]){
  const t=create(online);await t.filter('unreviewed');assert.equal(t.get('index'),0);
  await t.node('next').onclick();assert.equal(t.get('index'),3,'Skip approved and excluded images');
  await t.node('prev').onclick();assert.equal(t.get('index'),0);
  await t.node('approve').onclick();assert.equal(t.get('index'),3,'Approval advances within unreviewed filter');
  await t.node('approve').onclick();assert.equal(t.get('hasImage'),false,'Last pending approval shows empty view');
  assert.equal(t.node('canvas').hidden,true);assert.equal(t.node('approve').disabled,true);assert.equal(t.node('review-filter').disabled,false);
  assert.match(t.node('status').textContent,/Không còn ảnh chưa duyệt/);
  assert(t.requests.every(r=>r.cursor>=0),'Never save negative cursor on last approval');
  await t.filter('reviewed');assert.equal(t.get('hasImage'),true);assert.equal(t.node('approve').disabled,false);assert.equal(t.node('canvas').hidden,false);
  await t.filter('all');assert.equal(t.get('reviewFilter'),'all');assert.equal(t.get('rows.filter(matchesFilter).length'),3);
 }
 const failed=create(true);failed.state.fail=true;await failed.filter('unreviewed');assert.equal(failed.get('reviewFilter'),'all');assert.equal(failed.node('review-filter').value,'all');assert.match(failed.node('save-status').textContent,/CHƯA LƯU/);
 const empty=create(false);empty.get('rows.forEach(row=>row.reviewed=false)');await empty.filter('reviewed');assert.equal(empty.get('hasImage'),false);await empty.filter('all');assert.equal(empty.get('hasImage'),true);
 console.log('Filter: all/pending/approved, skip excluded, save before switch, last approval, empty recovery and conflict handling pass');
})().catch(error=>{console.error(error);process.exitCode=1});
