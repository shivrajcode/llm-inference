const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ids = new Set([...fs.readFileSync('web/index.html', 'utf8').matchAll(/id="([^"]+)"/g)].map((match) => match[1]));
const nodes = new Map();
const node = () => ({value:'',textContent:'',hidden:false,disabled:false,dataset:{},classList:{toggle(){}},children:[],events:{},append(...v){this.children.push(...v)},replaceChildren(){this.children=[]},setAttribute(){},addEventListener(k,v){this.events[k]=v},focus(){},scrollHeight:100,scrollTop:0,clientHeight:100});
const document = {getElementById(id){assert.ok(ids.has(id), `Unknown DOM id: ${id}`);if(!nodes.has(id))nodes.set(id,node());return nodes.get(id)},createElement:node,querySelectorAll:()=>[]};
let mode = 'complete';
let request;
let trimmed = 0;
let cancelledReaders = 0;
let releasedReaders = 0;
const context = vm.createContext({document,Option:function(text,value){return {text,value}},TextDecoder,AbortController,console,fetch:async(path,options)=>{
  if(path==='/api/status')return {ok:true,json:async()=>({loaded:true,model:'test.gguf',context:16384,auto_context:true})};
  if(path==='/api/models')return {ok:true,json:async()=>({models:[]})};
  request = JSON.parse(options.body);
  assert.deepEqual(Object.keys(request), ['messages'], 'Inference settings must be automatic');
  if (mode === 'http-error') return {ok:false,status:409,json:async()=>({error:'Model is busy.'})};
  const payload = 'data: {"token":"<script>alert(1)</script>"}\r\n\r\ndata: {"token":" café"}\n\ndata: '+JSON.stringify({done:true,context:16384,history_trimmed:trimmed})+'\n\n';
  const bytes = new TextEncoder().encode(mode==='interrupted'?'data: {"token":"partial"}\n\n':payload);
  let index=0;
  return {ok:true,body:{getReader(){return {
    async read(){
      if(mode==='abort') {
        document.getElementById('stop').events.click();
        assert.equal(options.signal.aborted,true);
        throw Object.assign(new Error('Aborted'), {name:'AbortError'});
      }
      return index===bytes.length?{done:true}:{done:false,value:bytes.subarray(index,++index)};
    },async cancel(){cancelledReaders++},releaseLock(){releasedReaders++}
  }}}};
}});
vm.runInContext(fs.readFileSync('web/app.js','utf8'),context);
const submit = async (text) => {
  document.getElementById('prompt').value=text;
  await document.getElementById('composer').events.submit({preventDefault(){}});
};
(async()=>{
  await new Promise(setImmediate);
  assert.equal(ids.has('max-tokens'),false);
  assert.equal(ids.has('temperature'),false);
  assert.equal(ids.has('context'),false);
  await submit('Hi');
  const messages=document.getElementById('messages').children;
  assert.equal(messages[1].children[1].textContent,'<script>alert(1)</script> café');
  assert.equal(document.getElementById('generation-status').textContent,'');
  assert.equal(vm.runInContext('history.length',context),2);
  assert.equal(request.messages.length,1);

  trimmed=2;
  await submit('Next question');
  assert.equal(vm.runInContext('history.length',context),2,'Trim old turns from future requests');
  assert.equal(vm.runInContext('history[0].content',context),'Next question');
  assert.match(document.getElementById('generation-status').textContent,/Earlier messages/);
  assert.equal(document.getElementById('messages').children.length,4,'Keep visible transcript');

  mode='interrupted';
  await submit('Again');
  assert.match(document.getElementById('notice').textContent,/stream ended early/);
  assert.equal(vm.runInContext('history.length',context),2);
  assert.equal(document.getElementById('prompt').value,'Again');
  assert.equal(document.getElementById('stop').hidden,true);

  mode='abort';
  await submit('Stop this');
  assert.equal(document.getElementById('generation-status').textContent,'Generation stopped.');
  assert.equal(document.getElementById('prompt').value,'Stop this');
  assert.equal(vm.runInContext('history.length',context),2);
  assert.equal(document.getElementById('stop').hidden,true);
  assert.equal(document.getElementById('send').disabled,false);

  mode='http-error';
  await submit('Retry later');
  assert.equal(document.getElementById('notice').textContent,'Model is busy.');
  assert.equal(document.getElementById('prompt').value,'Retry later');
  assert.equal(vm.runInContext('history.length',context),2);

  mode='complete';trimmed=0;
  vm.runInContext("history = Array.from({length:256}, (_, i) => ({role: i%2 ? 'assistant' : 'user', content: String(i)}))",context);
  await submit('Long conversation');
  assert.equal(request.messages.length,255);
  assert.equal(request.messages[0].content,'2','Drop complete oldest pair at API limit');
  assert.match(document.getElementById('generation-status').textContent,/Earlier messages/);
  assert.equal(cancelledReaders,releasedReaders);
  assert.equal(cancelledReaders,5);
  console.log('PASS: automatic settings payload, byte-split UTF-8/CRLF SSE, safe text rendering, history trimming, interrupted streams, abort, HTTP errors, API count bound and cleanup');
})().catch(e=>{console.error(e);process.exitCode=1});
