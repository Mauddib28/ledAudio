# Task Master Integration

This project uses Task Master for AI-driven development. The task management system helps organize and track the development of features through a structured workflow.

## Getting Started

The project has been initialized with Task Master. Here's how to use it:

### Project Structure

- `tasks.json`: Main task database file containing all task definitions
- `tasks/`: Directory containing individual task files for easy reference
- `scripts/`: Contains the PRD and planning documents
- `.cursor/rules/dev_workflow.mdc`: Cursor integration for Task Master

### Using Task Master with Cursor

When using Cursor AI, you can use natural language to interact with Task Master:

```
What tasks are available to work on next?
I'd like to implement task 2. What does it involve?
Task 3 is complete. Please update its status.
```

### Manual Task Management

You can also manage tasks manually using the CLI:

```bash
# List all tasks
task-master list

# Show the next task to work on
task-master next

# Set a task's status
task-master set-status --id=<id> --status=<status>

# Break down a task into subtasks
task-master expand --id=<id> --num=<number_of_subtasks>
```

## Task Workflow

1. **Task Discovery**: Find the next task to work on
2. **Implementation**: Use the task details as a guide
3. **Verification**: Test according to the task's test strategy
4. **Completion**: Mark the task as done

## API Keys

The Task Master integration uses Anthropic and Perplexity API keys. Update the `.cursor/mcp.json` file with your API keys to enable full functionality. 