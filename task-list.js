const fs = require('fs');

// Read the tasks.json file
const tasksData = JSON.parse(fs.readFileSync('tasks.json', 'utf8'));
const tasks = tasksData.tasks;

// Display project information
console.log(`\n=== ${tasksData.metadata.projectName} ===`);
console.log(`${tasksData.metadata.description}`);
console.log(`Version: ${tasksData.metadata.version}\n`);

// Display tasks in a table format
console.log('ID  | Title                            | Priority | Status | Dependencies');
console.log('----+----------------------------------+----------+--------+------------');

tasks.forEach(task => {
  const id = String(task.id).padStart(2, '0');
  const title = task.title.padEnd(32, ' ').substring(0, 32);
  const priority = task.priority.padEnd(8, ' ');
  const status = task.status.padEnd(6, ' ');
  const deps = task.dependencies.length > 0 ? task.dependencies.join(', ') : '-';
  
  console.log(`${id}  | ${title} | ${priority} | ${status} | ${deps}`);
});

console.log('\nTo view task details, check the individual task files in the tasks/ directory.');
console.log('To update task status, edit the tasks.json file directly.\n'); 