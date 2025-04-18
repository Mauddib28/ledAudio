const fs = require('fs');

// Read the existing tasks.json
const tasksJSON = JSON.parse(fs.readFileSync('tasks.json', 'utf8'));

// Define tasks based on the PRD
const tasks = [
  {
    id: 1,
    title: 'Implement Audio Input Processing',
    description: 'Capture and process audio input from microphone or line-in',
    details: 'Create modules for audio capture, FFT analysis, and feature extraction. Support both microphone and line-in inputs.',
    priority: 'high',
    status: 'todo',
    dependencies: [],
    testStrategy: 'Unit tests for audio processing functions, manual testing with sample audio files'
  },
  {
    id: 2,
    title: 'Develop LED Control System',
    description: 'Build system to control LED strips with different patterns',
    details: 'Create LED driver module, pattern generators, and mapping between audio features and visual parameters.',
    priority: 'high',
    status: 'todo',
    dependencies: [1],
    testStrategy: 'Test patterns with mock audio input, verify timing and responsiveness'
  },
  {
    id: 3,
    title: 'Create Basic User Interface',
    description: 'Implement simple controls for the system',
    details: 'Create interface for selecting visualization modes and adjusting parameters like brightness and sensitivity.',
    priority: 'medium',
    status: 'todo',
    dependencies: [2],
    testStrategy: 'User testing with simple tasks, verify all controls work as expected'
  },
  {
    id: 4,
    title: 'Hardware Integration',
    description: 'Integrate with Raspberry Pi and Pico W hardware',
    details: 'Ensure software works properly on target hardware, optimize for performance.',
    priority: 'medium',
    status: 'todo',
    dependencies: [1, 2],
    testStrategy: 'Test on actual hardware, measure latency and resource usage'
  },
  {
    id: 5,
    title: 'Enhance Visualization Patterns',
    description: 'Develop additional visualization modes and effects',
    details: 'Create more complex and visually appealing patterns based on audio characteristics.',
    priority: 'low',
    status: 'todo',
    dependencies: [2],
    testStrategy: 'Visual testing with various music genres, gather feedback'
  }
];

// Update tasks in the JSON
tasksJSON.tasks = tasks;

// Write back to the file
fs.writeFileSync('tasks.json', JSON.stringify(tasksJSON, null, 2));

console.log('Tasks generated successfully!'); 