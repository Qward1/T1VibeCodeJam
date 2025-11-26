const fs = require("fs");
const path = require("path");

function loadUserModule(codePath) {
  const mod = require(codePath);
  return mod;
}

function main() {
  const workdir = "/home/runner/app";
  const testsPath = path.join(workdir, "tests.json");
  const codePath = path.join(workdir, "user_code.js");
  const result = { tests: [], error: null };
  try {
    const data = JSON.parse(fs.readFileSync(testsPath, "utf-8"));
    const fnName = data.function_name;
    const cases = data.tests || [];
    const mod = loadUserModule(codePath);
    const fn = mod[fnName];
    if (typeof fn !== "function") throw new Error("Function not found: " + fnName);
    cases.forEach((c, idx) => {
      try {
        const out = fn.apply(null, c.input || []);
        result.tests.push({
          name: c.name || `test_${idx + 1}`,
          passed: JSON.stringify(out) === JSON.stringify(c.expected),
          output: out,
          expected: c.expected,
        });
      } catch (err) {
        result.tests.push({
          name: c.name || `test_${idx + 1}`,
          passed: false,
          error: err.message,
          trace: err.stack,
        });
      }
    });
  } catch (err) {
    result.error = err.message;
  }
  console.log(JSON.stringify(result));
}

main();
