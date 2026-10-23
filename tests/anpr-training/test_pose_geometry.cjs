// Synthetic geometry tests on an HD image displayed at half size.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../../scripts/test-tools/Yolo11VietnamPlatePoseTraining/review_pose.html'),'utf8');
const row=id=>({id,image:`images/${id}.png`,aliases:[id],points:[[.1,.1],[.2,.1],[.2,.2],[.1,.2]],group:'source',reviewer:'tester',reviewed:true,group_reviewed:true,exclude:false,proposal:'test'});
const nodes=new Map(),context=new Proxy({},{get:()=>()=>{}});
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',getContext:()=>context,getBoundingClientRect:()=>({left:0,top:0,width:960,height:540}),click(){}});return nodes.get(id)}
const sandbox={document:{getElementById:node,createElement:()=>({click(){}})},Image:class{naturalWidth=1920;naturalHeight=1080;set src(_value){this.onload()}},setTimeout:()=>1,clearTimeout:()=>{},alert:()=>{},URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{}};
vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*)<\/script>/)[1].replace('/*MANIFEST*/null',JSON.stringify({records:[row('a'),row('b')]})),sandbox);
const points=()=>JSON.parse(vm.runInContext('JSON.stringify(rows[index].points)',sandbox));
const event=(x,y)=>({clientX:x,clientY:y,pointerId:1,button:0});
function mode(value){node('geometry-mode').value=value;node('geometry-mode').onchange()}
function drag(x0,y0,x1,y1){node('canvas').onpointerdown(event(x0,y0));node('canvas').onpointermove(event(x1,y1));node('canvas').onpointerup(event(x1,y1))}
function boxValid(p){assert.equal(p.length,4);assert(p.flat().every(v=>v>=0&&v<=1));assert(p[0][0]<p[1][0]&&p[0][1]<p[3][1]);assert.equal(p[0][1],p[1][1]);assert.equal(p[1][0],p[2][0]);assert.equal(p[2][1],p[3][1]);assert.equal(p[0][0],p[3][0])}
mode('rectangle');const original=points();node('canvas').onpointerdown(event(30,40));node('canvas').onpointermove(event(130,100));assert.deepEqual(points(),original,'Preview must not modify saved points');node('canvas').onpointerup(event(130,100));let p=points();boxValid(p);assert(Math.abs((p[1][0]-p[0][0])*1920-200)<1e-8);assert(Math.abs((p[3][1]-p[0][1])*1080-120)<1e-8);assert.equal(vm.runInContext('rows[0].reviewed',sandbox),false);
node('canvas').onclick(event(500,500));assert.deepEqual(points(),p,'Synthetic click after drag is ignored');
mode('square');for(const coordinates of [[100,100,300,180],[300,300,100,180],[100,300,300,180],[300,100,100,180],[900,500,1200,800]]){drag(...coordinates);p=points();boxValid(p);assert(Math.abs((p[1][0]-p[0][0])*1920-(p[3][1]-p[0][1])*1080)<1e-8,'Square sides equal in source pixels')}
p=points();drag(200,200,200,200);assert.deepEqual(points(),p,'Zero drag leaves corners intact');node('canvas').onpointerdown(event(20,20));node('canvas').onpointermove(event(200,200));node('canvas').onpointercancel();assert.deepEqual(points(),p,'Cancel keeps old corners');
node('canvas').onpointerdown(event(20,20));node('next').onclick();const second=points();node('canvas').onpointerup(event(200,200));assert.deepEqual(points(),second,'Stale drag cannot alter next image');
mode('points');for(const [x,y]of [[20,20],[80,20],[80,80],[20,80]])node('canvas').onclick(event(x,y));assert.equal(points().length,4);
console.log('Geometry: rectangle, source-pixel square, four drag directions, bounds, preview/cancel, stale drag and point mode pass');
