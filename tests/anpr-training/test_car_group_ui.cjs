// Synthetic DOM and API: validate navigation waits for successful persistence.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
function element(){return {value:'',textContent:'',disabled:false,classList:{toggle(){}},replaceChildren(){},append(){},addEventListener(){},removeAttribute(){}}}
const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
node('filter').value='pending';
const records=Array.from({length:3},(_,i)=>({id:'synthetic'+i,plate:'30A1234'+i,manual_group:'',reviewer:'',approved:false,version:0,candidates:[],crop:'/crop/test.jpg',image:'/image/test.jpg'}));
const writes=[];let fail=false;
const context=vm.createContext({document:{getElementById:node,createElement:element,addEventListener(){}},window:{addEventListener(){}},localStorage:{getItem(){return 'Tester'},setItem(){}},fetch:async(url,opts)=>{
 if(url==='/api/state')return {ok:true,json:async()=>({records:structuredClone(records)})};
 const body=JSON.parse(opts.body);writes.push(body);
 return fail?{ok:false,json:async()=>({error:'Synthetic revision conflict'})}:{ok:true,json:async()=>({version:body.version+1})};
}});
const source=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_car_groups.html'),'utf8').split('<script>')[1].split('</script>')[0];
vm.runInContext(source,context);
(async()=>{
 await new Promise(setImmediate);
 assert.equal(vm.runInContext('current',context),'synthetic0');
 node('plate').value='30A99999';vm.runInContext('dirty=true',context);
 await vm.runInContext('go("synthetic1")',context);
 assert.equal(writes[0].plate,'30A99999');assert.equal(writes[0].approved,false);
 assert.equal(vm.runInContext('current',context),'synthetic1');
 fail=true;node('plate').value='30A88888';vm.runInContext('dirty=true',context);
 await vm.runInContext('go("synthetic2")',context);
 assert.equal(vm.runInContext('current',context),'synthetic1');assert.match(node('status').textContent,/conflict/);
 fail=false;await vm.runInContext('approve()',context);
 assert.equal(writes.at(-1).approved,true);assert.equal(vm.runInContext('current',context),'synthetic2');
 assert.equal(vm.runInContext('rows[1].approved',context),true);
 console.log('Car group UI: save, navigation, conflict and approval passed');
})().catch(e=>{console.error(e);process.exitCode=1});
