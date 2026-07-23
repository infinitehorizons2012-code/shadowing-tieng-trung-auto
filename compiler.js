const fs = require('fs');
const path = require('path');

const inputFile = path.join(__dirname, 'linguistic_analysis.json');
const outputJsFile = path.join(__dirname, 'src', 'lessonData.js');

if (!fs.existsSync(inputFile)) {
    console.error("linguistic_analysis.json not found!");
    process.exit(1);
}

const data = JSON.parse(fs.readFileSync(inputFile, 'utf-8'));

// If a src directory exists (e.g. Vite/React template), inject data as a JS module
if (fs.existsSync(path.join(__dirname, 'src'))) {
    const jsContent = `export const LESSON_DATA = ${JSON.stringify(data, null, 2)};`;
    fs.writeFileSync(outputJsFile, jsContent, 'utf-8');
    console.log("Injected data into src/lessonData.js");
} else {
    // If not, just create a standalone JS file that defines window.LESSON_DATA
    const jsContent = `window.LESSON_DATA = ${JSON.stringify(data, null, 2)};`;
    fs.writeFileSync(path.join(__dirname, 'lessonData.js'), jsContent, 'utf-8');
    console.log("Created lessonData.js (window.LESSON_DATA)");
}
