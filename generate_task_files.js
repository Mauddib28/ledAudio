const fs = require('fs');

// Read the tasks.json file
const tasksData = JSON.parse(fs.readFileSync('tasks.json', 'utf8'));
const tasks = tasksData.tasks;

// Create the tasks directory if it doesn't exist
if (!fs.existsSync('tasks')) {
  fs.mkdirSync('tasks');
}

// Generate individual task files
tasks.forEach(task => {
  const filename = `tasks/task_${String(task.id).padStart(3, '0')}.txt`;
  
  // Format the task content
  const content = `# Task ${task.id}: ${task.title}

## Description
${task.description}

## Details
${task.details}

## Priority
${task.priority}

## Status
${task.status}

## Dependencies
${task.dependencies.length > 0 ? task.dependencies.join(', ') : 'None'}

## Test Strategy
${task.testStrategy}
`;

  // Write the file
  fs.writeFileSync(filename, content);
  console.log(`Generated ${filename}`);
});

console.log('All task files generated successfully!'); 