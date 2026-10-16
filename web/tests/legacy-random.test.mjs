import test from 'node:test';
import assert from 'node:assert/strict';
import {legacyGaussianGenerator,legacyNoiseData} from '../simulation.js';

test('historical Rand48 / polar Box–Muller agrees with archived float C++ draws',()=>{
  const gaussian=legacyGaussianGenerator(12345);
  const expected=[.669526279,-.873199224,.445691675,.254597098,.923034787,-.392144114,-1.29932761,-.360336542,.258115619,1.96545434,-1.39727676,-.657735646];
  for(const value of expected)assert.ok(Math.abs(gaussian()-value)<1e-6);
});
test('legacy draws skip DC, pair complex h0 in the historical order, and remain reproducible',()=>{
  const data=legacyNoiseData(16,12345),g=legacyGaussianGenerator(12345);
  assert.deepEqual([...data.slice(0,4)],[0,0,0,0]);
  const a=g(),b=g(),c=g(),d=g();
  assert.deepEqual([...data.slice(4,8)],[c,-d,a,b]);
  assert.deepEqual(data,legacyNoiseData(16,12345));
  assert.notDeepEqual(data,legacyNoiseData(16,54321));
});
