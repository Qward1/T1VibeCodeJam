#!/bin/bash
set -e
cd /home/runner/app
g++ -std=c++17 -O2 -pipe -static -s user_code.cpp -o app.out
if [ ! -f tests.json ]; then
  echo '{"error":"tests.json not found"}'
  exit 1
fi
node -e "const fs=require('fs');const tests=JSON.parse(fs.readFileSync('tests.json','utf-8'));const { spawnSync } = require('child_process');const results={tests:[],error:null};tests.tests.forEach((t,idx)=>{const proc=spawnSync('./app.out',[],{input:JSON.stringify(t.input),encoding:'utf-8'});if(proc.status!==0){results.tests.push({name:t.name||('test_'+(idx+1)),passed:false,error:proc.stderr||proc.stdout});}else{let out=proc.stdout.trim();try{out=JSON.parse(out);}catch(e){}results.tests.push({name:t.name||('test_'+(idx+1)),passed:JSON.stringify(out)===JSON.stringify(t.expected),output:out,expected:t.expected});}});console.log(JSON.stringify(results));"
